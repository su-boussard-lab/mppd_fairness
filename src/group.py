from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

MIN_GROUP_SIZE = 10    

def calculate_tpr_and_fpr(y_true, y_pred, group_mask):
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for a given group.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        group_mask: Boolean mask indicating which samples belong to the group
        
    Returns:
        tuple: (TPR, FPR)
    """
    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask], labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr


def calculate_classification_metrics(y_true, y_pred, group_mask=None):
    """
    Calculate confusion matrix-based metrics: TPR, FPR, Sensitivity, and Specificity.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        group_mask: Boolean mask indicating which samples belong to the group.
                   If None, calculate for all samples.
        
    Returns:
        dict: Dictionary containing TPR, FPR, Sensitivity, and Specificity
    """
    if group_mask is None:
        group_mask = np.ones_like(y_true, dtype=bool)
        
    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask], labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'TPR': tpr,
        'Sensitivity': tpr,
        'FPR': fpr,
        'Specificity': specificity
    }


def calculate_demographic_parity_metrics(y_pred, demographics, group_mapping=None, min_group_size=MIN_GROUP_SIZE):
    """
    Calculate Demographic Parity metrics across demographic groups.
    
    Args:
        y_pred: Predicted labels
        demographics: Array of demographic values
        group_mapping: Optional dictionary mapping group codes to readable names
        min_group_size: Minimum number of samples required for a group
        
    Returns:
        dict: Dictionary containing DPD and DPR metrics
    """
    unique_groups = np.unique(demographics)
    positive_rates = {}
    
    # Calculate positive prediction rate for each group
    for group in unique_groups:
        if group not in group_mapping and group_mapping is not None:
            continue
            
        group_mask = demographics == group
        
        if np.sum(group_mask) < min_group_size:
            continue
            
        positive_rate = np.mean(y_pred[group_mask] == 1)
        group_name = group_mapping.get(group, f"Group {group}")
        positive_rates[group_name] = positive_rate
    
    # Calculate overall positive prediction rate
    overall_positive_rate = np.mean(y_pred == 1)
    
    # Calculate Demographic Parity Difference (DPD) and Ratio (DPR)
    dpd_values = {}
    dpr_values = {}
    
    for group_name, rate in positive_rates.items():
        # DPD: difference between group's positive rate and overall positive rate
        dpd_values[group_name] = rate - overall_positive_rate
        
        # DPR: ratio of group's positive rate to overall positive rate
        dpr_values[group_name] = rate / overall_positive_rate if overall_positive_rate > 0 else float('inf')
    
    max_dpd = max(abs(dpd) for dpd in dpd_values.values()) if dpd_values else 0
    avg_dpd = np.mean([abs(dpd) for dpd in dpd_values.values()]) if dpd_values else 0
    max_dpr_deviation = max(max(dpr, 1/dpr) for dpr in dpr_values.values() if dpr > 0) if dpr_values else 1
    
    # Calculate pairwise DPD (max difference between any two groups)
    pairwise_dpd = max(positive_rates.values()) - min(positive_rates.values()) if positive_rates else 0

    return {
        'Positive Rates': positive_rates,
        'Overall Positive Rate': overall_positive_rate,
        'DPD Values': dpd_values,
        'DPR Values': dpr_values,
        'Max DPD': max_dpd,
        'Avg DPD': avg_dpd,
        'Pairwise DPD': pairwise_dpd,
        'Max DPR Deviation': max_dpr_deviation,
    }


def calculate_group_metrics(y_true, y_pred, group_mask, overall_tpr, overall_fpr, overall_error_rate):
    """
    Calculate fairness metrics for a specific demographic group.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        group_mask: Boolean mask indicating which samples belong to the group
        overall_tpr: Overall true positive rate
        overall_fpr: Overall false positive rate
        overall_error_rate: Overall error rate
        
    Returns:
        dict: Dictionary containing metrics for the group
    """
    tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred, group_mask)
    
    group_positives = np.sum(y_pred[group_mask] == 1)
    precision = np.sum((y_true[group_mask] == 1) & (y_pred[group_mask] == 1)) / group_positives if group_positives > 0 else 0
    
    group_error_rate = np.mean(y_true[group_mask] != y_pred[group_mask])
    
    # Calculate fairness metrics
    eod = tpr - overall_tpr  # Equal Opportunity Difference
    eod_fpr = fpr - overall_fpr  # FPR difference
    avg_eod = (abs(eod) + abs(eod_fpr)) / 2  # Average Equalized Odds Difference
    
    # Calculate EDDI
    eddi = abs(group_error_rate - overall_error_rate) / max(overall_error_rate, 1 - overall_error_rate)

    classification_metrics = calculate_classification_metrics(y_true, y_pred, group_mask)
    
    return {
        'Size': np.sum(group_mask),
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision,
        'Error Rate': group_error_rate,
        'Sensitivity': classification_metrics['Sensitivity'],
        'Specificity': classification_metrics['Specificity'],
        'EOD': eod,
        'EOD (FPR)': eod_fpr,
        'Equalized Odds': avg_eod,
        'EDDI': eddi
    }

def calculate_summary_metrics(group_metrics, classification_metrics, overall_tpr, overall_fpr, overall_error_rate, dp_metrics, unique_groups, demographics, y_pred):
    """
    Calculate summary fairness metrics across all demographic groups.
    
    Args:
        group_metrics: Dictionary of metrics for each group
        classification_metrics: Overall classification metrics
        overall_tpr: Overall true positive rate
        overall_fpr: Overall false positive rate
        overall_error_rate: Overall error rate
        dp_metrics: Demographic parity metrics
        unique_groups: Array of unique demographic groups
        demographics: Array of demographic values
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing summary metrics
    """
    return {
        'Overall TPR': overall_tpr,
        'Overall FPR': overall_fpr,
        'Overall Error Rate': overall_error_rate,
        'Overall Sensitivity': classification_metrics['Sensitivity'],
        'Overall Specificity': classification_metrics['Specificity'],
        'Max EOD (TPR)': max(abs(m['EOD']) for m in group_metrics.values()),
        'Max EOD (FPR)': max(abs(m['EOD (FPR)']) for m in group_metrics.values()),
        'Max Equalized Odds': max(abs(m['Equalized Odds']) for m in group_metrics.values()),
        'Avg EOD (TPR)': np.mean([abs(m['EOD']) for m in group_metrics.values()]),
        'Avg EOD (FPR)': np.mean([abs(m['EOD (FPR)']) for m in group_metrics.values()]),
        'Avg Equalized Odds': np.mean([m['Equalized Odds'] for m in group_metrics.values()]),
        'Max EDDI': max(m['EDDI'] for m in group_metrics.values()),
        'Avg EDDI': np.mean([m['EDDI'] for m in group_metrics.values()]),
        'Demographic Parity': dp_metrics['Pairwise DPD'],
        'Max DPD': dp_metrics['Max DPD'],
        'Avg DPD': dp_metrics['Avg DPD'],
        'Max DPR Deviation': dp_metrics['Max DPR Deviation']
    }

def calculate_group_fairness_metrics(y_true, y_pred, demographics, group_mapping=None, min_group_size=MIN_GROUP_SIZE):
    """
    Calculate comprehensive fairness metrics for each demographic group.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        demographics: Array of demographic values
        group_mapping: Optional dictionary mapping group codes to readable names
        min_group_size: Minimum number of samples required for a group
        
    Returns:
        tuple: (group_metrics, summary_metrics)
    """
    # Calculate overall metrics
    overall_error_rate = np.mean(y_true != y_pred)
    overall_tpr, overall_fpr = calculate_tpr_and_fpr(y_true, y_pred, np.ones_like(y_true, dtype=bool))
    classification_metrics = calculate_classification_metrics(y_true, y_pred, np.ones_like(y_true, dtype=bool))
    
    unique_groups = np.unique(demographics)
    group_metrics = {}
    
    if group_mapping is None:
        raise ValueError("A group mapping must be provided to interpret demographic values.")
    
    # Calculate metrics for each group
    for group in unique_groups:
        if group not in group_mapping and group_mapping is not None:
            continue
            
        group_name = group_mapping.get(group, f"Group {group}")
        group_mask = demographics == group
        
        if np.sum(group_mask) < min_group_size:
            continue
        
        group_metrics[group_name] = calculate_group_metrics(
            y_true, y_pred, group_mask, overall_tpr, overall_fpr, overall_error_rate
        )
    
    dp_metrics = calculate_demographic_parity_metrics(y_pred, demographics, group_mapping, min_group_size)
    
    summary_metrics = calculate_summary_metrics(
        group_metrics, classification_metrics, overall_tpr, overall_fpr, overall_error_rate, 
        dp_metrics, unique_groups, demographics, y_pred
    )
    
    return group_metrics, summary_metrics


def visualize_fairness_metrics(fairness_metrics, low_threshold=0.01, medium_threshold=0.05, 
                              high_threshold=0.1, figsize_bar=(12, 8), figsize_radar=(10, 10),
                              title=None, show_plot="both", save_pdf=False, pdf_filename=None):
    """
    Visualize fairness metrics using bar chart and radar chart.
    """
    thresholds = {
        'low': low_threshold,
        'medium': medium_threshold,
        'high': high_threshold
    }
    
    df_metrics = pd.DataFrame({
        'Metric': list(fairness_metrics.keys()),
        'Value': list(fairness_metrics.values())
    })
    
    def interpret_value(value):
        if value < thresholds['low']:
            return 'Low Disparity'
        elif value < thresholds['medium']:
            return 'Moderate Disparity'
        elif value < thresholds['high']:
            return 'High Disparity'
        else:
            return 'Very High Disparity'
    
    df_metrics['Interpretation'] = df_metrics['Value'].apply(interpret_value)
    
    if show_plot == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize_bar[0] + figsize_radar[0], max(figsize_bar[1], figsize_radar[1])))
    elif show_plot == "bar":
        fig, ax1 = plt.subplots(figsize=figsize_bar)
    elif show_plot == "radar":
        fig, ax2 = plt.subplots(figsize=figsize_radar, subplot_kw={'polar': True})
    
    if show_plot in ["both", "bar"]:
        sns.barplot(x='Value', y='Metric', data=df_metrics, 
                    palette={'Low Disparity': 'green', 
                             'Moderate Disparity': 'yellow',
                             'High Disparity': 'orange',
                             'Very High Disparity': 'red'},
                    hue='Interpretation', ax=ax1)
        
        for i, patch in enumerate(ax1.patches):
            patch.set_height(0.65)
            metric_idx = i % len(df_metrics)
            patch.set_y(metric_idx - 0.325)
        
        ax1.axvline(x=thresholds['low'], color='green', linestyle='--', alpha=0.7)
        ax1.axvline(x=thresholds['medium'], color='orange', linestyle='--', alpha=0.7)
        ax1.axvline(x=thresholds['high'], color='red', linestyle='--', alpha=0.7)
        
        ax1.text(thresholds['low'], -0.5, f"Low: {thresholds['low']}", color='green')
        ax1.text(thresholds['medium'], -0.5, f"Medium: {thresholds['medium']}", color='orange')
        ax1.text(thresholds['high'], -0.5, f"High: {thresholds['high']}", color='red')
        
        for i, v in enumerate(df_metrics['Value']):
            ax1.text(v + 0.001, i, f"{v:.4f}", va='center')
        
        ax1.set_title("Fairness Metrics" if title is None else title, fontsize=12, pad=15)
        ax1.set_xlabel('Disparity Value (closer to 0 is better)', fontsize=12)
        ax1.set_ylabel('Fairness Metric', fontsize=12)
    
    # Radar chart
    if show_plot in ["both", "radar"]:
        metrics = df_metrics['Metric'].tolist()
        values = df_metrics['Value'].tolist()
        
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        if show_plot == "both":
            ax2 = plt.subplot(1, 2, 2, polar=True)
        
        ax2.plot(angles, values, linewidth=2, linestyle='solid')
        ax2.fill(angles, values, alpha=0.25)
        
        for threshold_name, threshold_value in thresholds.items():
            circle_values = [threshold_value] * (N + 1)
            ax2.plot(angles, circle_values, linestyle='--', alpha=0.75, 
                    label=f"{threshold_name.capitalize()}: {threshold_value}")
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics, size=10)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax2.set_title(f'Radar View', size=12)
    
    plt.tight_layout()

    if save_pdf:
        plt.savefig(f"{pdf_filename}.pdf", dpi=300, bbox_inches='tight', format='pdf')
    
    plt.show()


def calculate_and_visualize_fairness_metrics(y_true, y_pred, demographic_variable, group_mapping, title=None, show_plot="both",
                                             save_pdf=False, pdf_filename=None):
    """
    Calculate fairness metrics for a demographic variable and visualize the results.
    """
    metrics = calculate_group_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        demographics=demographic_variable,
        group_mapping=group_mapping
    )
    
    def get_visualize_metrics(metrics_dict):
        metrics_list = ['Max DPD', 'Avg DPD',
                        'Max EOD (TPR)', 'Avg EOD (TPR)', 
                        'Max Equalized Odds', 'Avg Equalized Odds',
                        'Max EDDI', 'Avg EDDI']
        ret_dict = {}
        for metric in metrics_list:
            if metric not in metrics_dict:
                metrics_dict[metric] = 0
            ret_dict[metric] = metrics_dict[metric]
        
        return ret_dict
    
    visualize_metrics = get_visualize_metrics(metrics[1])
    
    visualize_fairness_metrics(
        visualize_metrics, 
        figsize_bar=(8, 6),
        figsize_radar=(7, 7),
        title=title,
        show_plot=show_plot,
        save_pdf=save_pdf,
        pdf_filename=pdf_filename
    )
    
    return metrics, visualize_metrics


def visualize_metric_by_group(group_metrics, metric_name, raw_value_metric=None, title=None, 
                             figsize=(10, 6), low_threshold=0.01, medium_threshold=0.05, 
                             high_threshold=0.1, save_pdf=False, pdf_filename=None):
    """
    Visualize a specific fairness metric across different demographic groups.
    
    Parameters:
    -----------
    group_metrics : dict
        Dictionary with group names as keys and metric dictionaries as values
    metric_name : str
        Name of the metric to visualize (e.g., 'EOD (FPR)')
    raw_value_metric : str, optional
        Name of a raw metric to display next to group size (e.g., 'FPR')
    title : str, optional
        Title for the visualization
    figsize : tuple, optional
        Figure size (width, height)
    low_threshold : float, optional
        Threshold for low disparity
    medium_threshold : float, optional
        Threshold for medium disparity
    high_threshold : float, optional
        Threshold for high disparity
        
    Returns:
    --------
    None (displays the plot)
    """
    groups = []
    values = []
    sizes = []
    raw_values = []
    raw_metric_values = []
    
    for group_name, metrics in group_metrics.items():
        if metric_name in metrics:
            groups.append(group_name)
            values.append(abs(metrics[metric_name]))
            raw_values.append(metrics[metric_name])
            sizes.append(metrics['Size'])
            
            if raw_value_metric and raw_value_metric in metrics:
                raw_metric_values.append(metrics[raw_value_metric])
            else:
                raw_metric_values.append(None)
    
    df = pd.DataFrame({
        'Group': groups,
        'Value': values,
        'RawValue': raw_values,
        'Size': sizes,
        'RawMetricValue': raw_metric_values
    })
    
    df = df.sort_values('Value', ascending=False)
    
    def interpret_value(value):
        if value < low_threshold:
            return 'Low Disparity'
        elif value < medium_threshold:
            return 'Moderate Disparity'
        elif value < high_threshold:
            return 'High Disparity'
        else:
            return 'Very High Disparity'
    
    df['Interpretation'] = df['Value'].apply(interpret_value)
    
    plt.figure(figsize=figsize)
    
    color_map = {
        'Low Disparity': 'green', 
        'Moderate Disparity': 'yellow',
        'High Disparity': 'orange',
        'Very High Disparity': 'red'
    }
    
    y_positions = np.arange(len(df))
    legend_handles = []
    legend_labels = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        color = color_map[row['Interpretation']]
        bar = plt.barh(y_positions[i], row['Value'], color=color, height=0.6)
        
        plt.text(row['Value'] + 0.001, y_positions[i], f"{row['Value']:.4f}", va='center')
        
        if row['Interpretation'] not in legend_labels:
            legend_handles.append(bar)
            legend_labels.append(row['Interpretation'])
    
    plt.yticks(y_positions, df['Group'])
    
    for i, (size, raw_value, raw_metric_value, pos) in enumerate(zip(df['Size'], df['RawValue'], df['RawMetricValue'], y_positions)):
        if raw_value_metric and raw_metric_value is not None:
            plt.text(-0.005, pos - 0.25, f"n={size}, {raw_value_metric}={raw_metric_value:.2f}", 
                     ha='right', va='center', fontsize=8, color='gray', style='italic')
        else:
            plt.text(-0.005, pos - 0.25, f"n={size}", ha='right', va='center', 
                     fontsize=8, color='gray', style='italic')
    
    plt.axvline(x=low_threshold, color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=medium_threshold, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=high_threshold, color='red', linestyle='--', alpha=0.7)
    
    plt.text(low_threshold, -0.5, f"Low: {low_threshold}", color='green')
    plt.text(medium_threshold, -0.5, f"Medium: {medium_threshold}", color='orange')
    plt.text(high_threshold, -0.5, f"High: {high_threshold}", color='red')
    
    plt.legend(legend_handles, legend_labels, loc='upper right')
    
    plt.xlabel(f'{metric_name} Value (closer to 0 is better)')
    plt.ylabel('Demographic Group')
    plt.title(title or f'{metric_name} by Demographic Group')
    
    plt.subplots_adjust(left=0.2)
    plt.tight_layout()
    
    if save_pdf:
        if pdf_filename is None:
            safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            pdf_filename = f"fairness_metric_{safe_metric_name}"
        plt.savefig(f"{pdf_filename}.pdf", dpi=300, bbox_inches='tight', format='pdf')
    
    plt.show()