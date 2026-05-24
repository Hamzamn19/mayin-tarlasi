#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUTS_DIR / "landmine_tabular_dataV3.csv"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
features = ['circularity', 'thermal_contrast', 'thermal_gradient']
df.dropna(subset=features + ['label'], inplace=True)

# Map labels
df['Class'] = df['label'].map({0: 'Background Clutter', 1: 'Landmine'})

# Style parameters for dark theme
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='none')
colors = {0: '#3b82f6', 1: '#f59e0b'} # blue = clutter, amber = landmine

feature_labels = {
    'circularity': 'Geometric Circularity (0 to 1)',
    'thermal_contrast': 'Thermal Contrast (Object - BG)',
    'thermal_gradient': 'Thermal Edge Gradient (Sobel)'
}

print("Plotting distributions...")
for ax, feat in zip(axes, features):
    ax.set_facecolor('none')
    # Plot background clutter
    sns.kdeplot(
        data=df[df['label'] == 0], x=feat, fill=True, 
        color=colors[0], alpha=0.35, linewidth=1.5, 
        label='Background Clutter', ax=ax
    )
    # Plot landmines
    sns.kdeplot(
        data=df[df['label'] == 1], x=feat, fill=True, 
        color=colors[1], alpha=0.45, linewidth=1.5, 
        label='Landmine', ax=ax
    )
    
    # Titles and labels styled for the dark theme
    ax.set_title(feature_labels[feat], color='#f8fafc', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('Density' if feat == 'circularity' else '', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    
    # Hide top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    ax.grid(True, linestyle=':', alpha=0.15, color='#cbd5e1')

# Unified legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, 
           facecolor='#090f24', edgecolor='#1e293b', fontsize=11)

plt.tight_layout()
# Adjust top space to prevent legend overlap
plt.subplots_adjust(top=0.80)

out_path = PLOTS_DIR / "feature_distributions_top3.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Success: Saved top 3 feature distribution plot to {out_path}")
