"""Sensitivity analysis and visualization utilities."""

from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import PARAM_DISPLAY_MAP


def run_sensitivity_analysis(
    model: nn.Module,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str],
    device: torch.device,
    batch_size: int = 4096
) -> pd.DataFrame:
    """
    Evaluate model performance across parameter values using batch processing.
    """
    model.eval()
    results = []
    
    print(f"Predicting on {len(test_df)} samples (batch size: {batch_size})...")
    
    X_raw = test_df[feature_cols].values
    all_probs = []
    
    # Process in batches to prevent RAM overflow
    for i in range(0, len(X_raw), batch_size):
        batch_raw = X_raw[i:i + batch_size]
        batch_scaled = scaler.transform(batch_raw)
        batch_tensor = torch.FloatTensor(batch_scaled).to(device)
        
        with torch.no_grad():
            batch_out = torch.sigmoid(model(batch_tensor))
            all_probs.append(batch_out.cpu().numpy().flatten())
        
        del batch_tensor
    
    # Concatenate all batch results
    global_probs = np.concatenate(all_probs)
    global_preds = (global_probs > 0.5).astype(int)
    
    analysis_df = test_df.copy()
    analysis_df['predicted'] = global_preds
    
    print("Calculating sensitivity metrics...")
    
    for col_name, display_name in PARAM_DISPLAY_MAP.items():
        if col_name not in analysis_df.columns:
            continue
        
        unique_vals = sorted(analysis_df[col_name].unique())
        
        for val in unique_vals:
            mask = analysis_df[col_name] == val
            
            if not mask.any():
                continue
            
            y_true = analysis_df.loc[mask, 'label'].values
            y_pred = analysis_df.loc[mask, 'predicted'].values
            
            results.append({
                'Parameter_Type': col_name,
                'Parameter_Label': display_name,
                'Value': val,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1_Score': f1_score(y_true, y_pred, zero_division=0),
                'Count': mask.sum()
            })
    
    return pd.DataFrame(results)


def plot_sensitivity_metric(
    sensitivity_df: pd.DataFrame,
    metric: str = 'F1_Score',
    save_path: str = None
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    params = ['epsilon', 'attacker_ratio', 'target_set_size']
    labels = ['$\\epsilon$', '$\\beta$', '$r$']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    line_color = '#1f77b4'
    
    for i, param in enumerate(params):
        ax = axes[i]
        data = sensitivity_df[sensitivity_df['Parameter_Type'] == param].copy()
        
        if len(data) == 0:
            ax.set_title(f"No data for {param}")
            continue
        
        # Ensure Value is numeric (may be object dtype when mixed with string params)
        data['Value'] = pd.to_numeric(data['Value'])
        
        ax.plot(
            data['Value'], data[metric],
            marker='X', markersize=8,
            linestyle='--', linewidth=2,
            color=line_color, label='Proposed DL Model'
        )
        
        ax.set_xlabel(labels[i], fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel(metric.replace('_', ' '), fontsize=14, fontweight='bold')
        
        ax.set_ylim(-0.05, 1.05)
        
        if param == 'target_set_size':
            ax.set_xticks(data['Value'].unique())
        
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f"Impact of Parameters on {metric.replace('_', ' ')}",
        fontsize=16, y=1.05
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig(f'sensitivity_{metric.lower()}.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: sensitivity_{metric.lower()}.png")
    
    plt.close(fig)

