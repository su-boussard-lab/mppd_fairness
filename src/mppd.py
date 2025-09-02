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