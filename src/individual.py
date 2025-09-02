from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

from .constants import (
    COMORBIDITY_FEATURES,
    IF_FEATURES, 
    CLINICAL_FEATURES,
)


def cosine_distance(X, baseline=0.1):
    """
    Calculate pairwise cosine distances between patients.
    
    Parameters:
    - X: DataFrame or array containing patient data
    
    Returns:
    - distances: A square matrix of pairwise cosine distances between patients
    """
    if hasattr(X, 'values'):
        data = X.values
    else:
        data = X
    
    data = data + baseline
    distances = squareform(pdist(data, metric='cosine'))
    
    return distances


def get_feature_subset(X, feature_set='all'):
    """
    Select a subset of features from the dataset based on the specified feature set.
    
    Parameters:
    - X: DataFrame containing patient data
    - feature_set: String indicating which feature set to use:
                  'all': All features except patient ID
                  'clinical': Only clinical features (age, Charlson_score, pre_pain_score, bmi)
                  'comorbidity': Only comorbidity features
    
    Returns:
    - X_subset: DataFrame containing only the selected features
    """
    if feature_set == 'all':
        return X.drop(['pat_deid'], axis=1) if 'pat_deid' in X.columns else X
    
    elif feature_set == 'select_clinical':
        return X[CLINICAL_FEATURES]
    
    elif feature_set == 'comorbidity':
        return X[COMORBIDITY_FEATURES]
    
    elif feature_set == 'all_clinical':
        return X[IF_FEATURES]
    
    else:
        raise ValueError(f"Unknown feature set: {feature_set}. Valid options are 'all', 'all_clinical', 'select_clinical',  or 'comorbidity'.")


def calculate_distance_based_fairness(X_test, predictions, distance_function=None, feature_set='all', distance_threshold=0.1, percentile=None):
    """
    Measure if similar patients get similar predictions
    
    Parameters:
    - X_test: feature matrix of test data
    - predictions: model predictions (probabilities or binary)
    - distance_function: function to calculate distances between patients
                        If None, will use comorbidity_distance by default
    - distance_threshold: fixed threshold to consider patients as similar (default: 0.1)
    - percentile: if provided, use this percentile of distances as threshold instead of fixed value
                 (e.g., 5 means use the 5th percentile of all distances)
    
    Returns:
    - avg_diff: average prediction difference between similar patients
    - num_similar_pairs: number of similar pairs found
    - threshold_used: the actual threshold used (either fixed or percentile-based)
    """
    X_subset = get_feature_subset(X_test, feature_set)

    if distance_function is None:
        distances = cosine_distance(X_subset)
    else:
        distances = distance_function(X_subset)
    
    if percentile is not None:
        flat_distances = distances[~np.eye(distances.shape[0], dtype=bool)]
        threshold_used = np.percentile(flat_distances, percentile)
        print(f"Using {percentile}th percentile as threshold: {threshold_used:.4f}")
    else:
        threshold_used = distance_threshold
        print(f"Using fixed threshold: {threshold_used:.4f}")
    
    similar_pairs = np.where(distances < threshold_used)
    
    not_same = similar_pairs[0] != similar_pairs[1]
    similar_pairs = (similar_pairs[0][not_same], similar_pairs[1][not_same])
    
    prediction_differences = abs(predictions[similar_pairs[0]] - predictions[similar_pairs[1]])
    
    num_similar_pairs = len(prediction_differences)
    if num_similar_pairs > 0:
        avg_diff = np.mean(prediction_differences)
        print(f"Found {num_similar_pairs} similar pairs")
        n_patients = len(X_test)
        avg_pairs_per_patient = num_similar_pairs / n_patients if n_patients > 0 else 0
        print(f"Average number of similar pairs per patient: {avg_pairs_per_patient:.2f}")    
        print(f"Average prediction difference: {avg_diff:.4f}")
        return avg_diff, num_similar_pairs, threshold_used
    else:
        print("No similar pairs found. Try increasing the threshold or percentile.")
        return None, 0, threshold_used
    

def analyze_individual_fairness_by_group(
    X_test, 
    y_pred_proba, 
    group_column, 
    feature_set='select_clinical',
    distance_function=cosine_distance,
    distance_threshold=0.05,
    max_samples=50,
    attribute_mappings=None,
    process_specific_group=None
):
    """
    Analyze individual fairness by demographic group.
    
    Parameters:
    -----------
    X_test : pandas.DataFrame
        Test data with features
    y_pred_proba : numpy.ndarray
        Predicted probabilities from the model
    group_column : str
        Column name for the demographic group to analyze
    feature_set : str, default='select_clinical'
        Feature set to use for distance calculation
    distance_function : function, default=cosine_distance
        Function to calculate distances between patients
    distance_threshold : float, default=0.05
        Threshold for considering patients similar
    max_samples : int, default=50
        Maximum number of samples to use from each group
    attribute_mappings : dict, default=None
        Mapping from numeric codes to group labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with fairness metrics by group
    dict
        Raw results dictionary with detailed statistics
    """
    X_test_mapped = X_test.copy()
    if attribute_mappings and group_column in attribute_mappings:
        X_test_mapped[group_column] = X_test_mapped[group_column].map(attribute_mappings[group_column])
    
    X_subset = get_feature_subset(X_test_mapped, feature_set)
    calculated_distance = distance_function(X_subset)
    df_index = X_test_mapped.index.tolist()
    pairwise_distance = pd.DataFrame(
        data=calculated_distance,
        index=df_index,
        columns=df_index
    )
    np.fill_diagonal(pairwise_distance.values, np.nan)
    
    result_dict = {}

    def process_patient_group(patient_group):
        print(f"Processing patient group: {patient_group}")
        reference_mask = X_test_mapped[group_column] == patient_group

        reference_patients = X_test[reference_mask]
        reference_size = len(reference_patients)

        print(f"Found {reference_size} patients in reference group '{patient_group}'")
        if reference_size > max_samples:
            reference_indices = np.random.choice(reference_size, max_samples, replace=False)
            reference_patients = reference_patients.iloc[reference_indices]
            print(f"Randomly sampled {max_samples} patients for analysis")

        reference_patient_indices = reference_patients.index.tolist()
        distance_subset = pairwise_distance.loc[reference_patient_indices, :]
        distance_subset_filtered = distance_subset.copy()
        mask = distance_subset_filtered < distance_threshold
        distance_subset_filtered = distance_subset_filtered.where(mask)

        distance_df = distance_subset_filtered.transpose().stack().reset_index()
        distance_df.columns = ['reference_idx', 'patient_idx', 'distance']
        distance_df = distance_df.dropna(subset=['distance'])
        print(f"Distance dataframe shape: {distance_df.shape}")
        
        distance_df['reference_group'] = X_test_mapped[group_column].loc[distance_df.reference_idx].reset_index(drop=True)
        y_proba_df = pd.DataFrame(y_pred_proba, index=X_test.index, columns=['y_pred_proba'])
        distance_df['patient_proba'] = y_proba_df.loc[distance_df.patient_idx].reset_index(drop=True)
        distance_df['reference_proba'] = y_proba_df.loc[distance_df.reference_idx].reset_index(drop=True)
        distance_df['proba_diff'] = np.abs(distance_df['patient_proba'] - distance_df['reference_proba'])
        
        return distance_df.groupby('reference_group')['proba_diff'].describe()
    
    if process_specific_group is None:
        for patient_group in X_test_mapped[group_column].unique():
            result_dict[patient_group] = process_patient_group(patient_group)
    else:
        result_dict[process_specific_group] = process_patient_group(process_specific_group)

    all_results = []
    for group, group_data in result_dict.items():
        for reference_group, stats in group_data.iterrows():
            row = {
                'group': group,
                'reference_group': reference_group,
                'count': stats['count'],
                'mean_proba_diff': stats['mean'],
                'std_proba_diff': stats['std'],
                'min_proba_diff': stats['min'],
                '25%_proba_diff': stats['25%'],
                'average_proba_diff': stats['50%'],
                '75%_proba_diff': stats['75%'],
                'max_proba_diff': stats['max']
            }
            all_results.append(row)

    result_df = pd.DataFrame(all_results)
    
    return result_df


def filter_fairness_results(result_df, exclude_groups=None, group_column='group', reference_column='reference_group'):
    """
    Filter fairness results to exclude specified groups.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame containing fairness results
    exclude_groups : list or None
        List of group values to exclude from both group and reference_group columns
    group_column : str
        Name of the column containing group values
    reference_column : str
        Name of the column containing reference group values
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with excluded groups removed
    """
    if exclude_groups is None:
        return result_df
    
    filtered_df = result_df.copy()
    
    for group in exclude_groups:
        filtered_df = filtered_df[
            (filtered_df[group_column] != group) & 
            (filtered_df[reference_column] != group)
        ]
    
    print(f"Filtered DataFrame shape: {filtered_df.shape}")
    print(f"Removed {result_df.shape[0] - filtered_df.shape[0]} rows with excluded groups: {exclude_groups}")
    
    return filtered_df



def plot_fairness_heatmap(
    result_df, 
    title, 
    figsize=(10, 8), 
    cmap='YlOrRd', 
    annot=True, 
    vmin=None, 
    vmax=None, 
    savefig=False, 
    savepath=None
):
    """
    Create a heatmap of mean probability differences between groups, sorted by total interaction count.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame containing fairness analysis results
    title : str
        Title for the heatmap
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    cmap : str, default='YlOrRd'
        Color map for the heatmap
    annot : bool, default=True
        Whether to annotate cells with numerical value
    vmin, vmax : float, optional
        Minimum and maximum values for color scaling. If None, uses data min/max.
    savefig : bool, default=False
        Whether to save the figure to a file
    savepath : str, optional
        Path to save the figure. If None and savefig is True, uses 'fairness_heatmap.png'
    """
    group_counts = pd.concat([
        result_df.groupby('group')['count'].sum(),
        result_df.groupby('reference_group')['count'].sum()
    ]).groupby(level=0).sum().to_dict()
    
    sorted_groups = sorted(group_counts.keys(), key=lambda x: group_counts[x], reverse=True)
    
    heatmap_data = result_df.pivot(
        index='reference_group',
        columns='group',
        values='mean_proba_diff'
    )
    
    heatmap_data = heatmap_data.reindex(
        index=sorted_groups[::-1], 
        columns=sorted_groups,
    )
    
    n = heatmap_data.shape[0]
    mask = np.tril(np.ones((n, n), dtype=bool), k=0)
    mask = np.flipud(mask)
    mask = ~mask
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt='.3f',
        cmap=cmap,
        center=None,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Mean Probability Difference'},
        mask=mask
    )
    
    plt.title(title, pad=20)
    plt.xlabel('Group')
    plt.ylabel('Reference Group')
    plt.tight_layout()
    
    if savefig:
        if savepath is None:
            savepath = "fairness_heatmap.pdf"
        plt.savefig(savepath, bbox_inches='tight', format='pdf', dpi=300)
        print(f"Figure saved to {savepath}")
    
    plt.show()


def save_fairness_results(results, save_dir='results/individual', suffix=''):
    os.makedirs(save_dir, exist_ok=True)
    
    date = datetime.now().strftime('%Y%m%d')
    
    filepath = os.path.join(save_dir, f'ind_results_{date}{suffix}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved all fairness results to: {filepath}")
    print(f"Number of DataFrames saved: {len(results)}")
    return filepath

def load_fairness_results(filepath):    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} DataFrames from: {filepath}")
    return results