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


def plot_multiple_group_comparison_proba_diff(result_dfs, sensitive_attributes, title=None, figsize=(18, 10), 
                                             low_threshold=0.025, medium_threshold=0.05, 
                                             high_threshold=0.1, save_path=None):
    """
    Create a dot plot showing average probability differences for multiple groups.
    
    Parameters:
    -----------
    result_dfs : list of pandas.DataFrame
        List of DataFrames containing the fairness results, each with columns:
        'group', 'reference_group', 'mean_proba_diff', 'count'
        Each DataFrame should have only one unique value in the 'group' column.
    title : str, optional
        Title for the plot. If None, a default title is used.
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    low_threshold : float, optional
        Threshold for low disparity (default: 0.025)
    medium_threshold : float, optional
        Threshold for medium disparity (default: 0.05)
    high_threshold : float, optional
        Threshold for high disparity (default: 0.1)
    save_path : str, optional
        If provided, the plot will be saved to this path.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object.
    """
    plt.figure(figsize=figsize)

    group_names = [df['group'].iloc[0] for df in result_dfs]
    group_positions = np.arange(len(result_dfs)) * 2

    all_reference_groups = set()
    for df in result_dfs:
        all_reference_groups.update(df['reference_group'].unique())
    all_reference_groups = sorted(list(all_reference_groups))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_reference_groups)))
    color_map = {ref_group: color for ref_group, color in zip(all_reference_groups, colors)}

    for i in range(1, len(result_dfs)):
        x_pos = (group_positions[i-1] + group_positions[i]) / 2
        plt.axvline(x=x_pos, color='lightgray', linestyle='-', alpha=0.5, zorder=0)

    for i, (result_df, group_name) in enumerate(zip(result_dfs, group_names)):
        ref_groups = result_df['reference_group'].values
        averages = result_df['mean_proba_diff'].values
        counts = result_df['count'].values
        
        same_group_idx = np.where(ref_groups == group_name)[0]
        
        for j, (ref_group, average, count) in enumerate(zip(ref_groups, averages, counts)):
            is_same_group = ref_group == group_name
            
            plt.scatter(
                group_positions[i], 
                average,
                s=1200 if is_same_group else 400,
                color=color_map[ref_group],
                edgecolor='black' if is_same_group else None,
                linewidth=3 if is_same_group else 0,
                alpha=0.8,
                zorder=10 if is_same_group else 5
            )
            
            plt.text(
                group_positions[i] + 0.2, 
                average,
                f"{ref_group}",
                va='center',
                fontsize=14,
                weight='bold' if is_same_group else 'normal',
                zorder=15
            )

    for i, (result_df, attr_name) in enumerate(zip(result_dfs, sensitive_attributes)):
        averages = result_df['mean_proba_diff'].values
        ref_groups = result_df['reference_group'].values
        
        min_idx = np.argmin(averages)
        max_idx = np.argmax(averages)
        min_val = averages[min_idx]
        max_val = averages[max_idx]
        min_group = ref_groups[min_idx]
        max_group = ref_groups[max_idx]
        
        diff = max_val - min_val
        line_width = 1 + 25 * (diff / 0.15)
        line_alpha = min(0.8, max(0.3, diff / 0.15))
        
        font_size = 10 + 6 * (diff / 0.15)
        font_size = min(16, max(10, font_size))
        
        plt.plot([group_positions[i], group_positions[i]], [min_val, max_val], 
                 color='darkred', linewidth=line_width, alpha=line_alpha, zorder=1)
        
        plt.text(group_positions[i] - 0.1, (min_val + max_val) / 2, 
                f"Δ = {diff:.3f}", ha='right', va='center', 
                fontsize=font_size, color='darkred', fontweight='bold')
    
    legend_elements = []
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                          markersize=37, markeredgecolor='black', markeredgewidth=1.5,
                          label='Same Group (Reference)'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                          markersize=21, label='Different Group (Comparison)'))
    
    legend = plt.legend(
        handles=legend_elements, 
        loc='upper right', 
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        borderpad=1.2,
        labelspacing=1.5,
        handletextpad=1.0,
        handlelength=3.0,
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.xlabel('Sensitive Attribute Group', fontsize=16)
    plt.ylabel('Mean Prediction Probability Difference', fontsize=16)
    
    if title is None:
        title = 'Mean Prediction Probability Differences Across Patient Groups'
    plt.title(title, fontsize=21)
    
    plt.xticks(group_positions, sensitive_attributes, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    all_averages = []
    for df in result_dfs:
        all_averages.extend(df['mean_proba_diff'].values)
    
    y_min = min(all_averages) - 0.01
    y_max = max(all_averages) + 0.01
    plt.ylim(y_min, y_max)

    plt.xlim(min(group_positions) - 1, max(group_positions) + 2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    
    plt.show()
    
    return plt.gcf()