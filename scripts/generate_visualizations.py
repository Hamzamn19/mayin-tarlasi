#!/usr/bin/env python3
"""
Generate professional visualizations for README from evaluation data.
Usage: python scripts/generate_visualizations.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SETUP
# ============================================
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Professional style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'

print("=" * 60)
print("📊 GENERATING VISUALIZATIONS FOR README")
print("=" * 60 + "\n")

# ============================================
# 1. LOAD EVALUATION RESULTS
# ============================================
try:
    eval_df = pd.read_csv(OUTPUTS_DIR / "hybrid_evaluation_results.csv")
    print(f"✅ Loaded: hybrid_evaluation_results.csv ({len(eval_df)} strategies)")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================
# 2. MODEL PERFORMANCE COMPARISON
# ============================================
print("\n🔧 Generating: Model Performance Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Model Performance Comparison: Precision vs Recall vs F1 Score', 
             fontsize=14, fontweight='bold', y=0.98)

labels = ['YOLO26\nOnly', 'YOLO26\n+ RF', 'YOLO26\n+ LR', 'Unanimous\nVoting', 'Majority\nVoting', 'Soft\nWeighted']
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D', '#A8E6CF', '#FFD3B6']

# Precision
for i, (prec, label) in enumerate(zip(eval_df['Precision']*100, labels)):
    axes[0].bar(i, prec, color=colors[i], alpha=0.85, edgecolor='black', linewidth=1.5)
    axes[0].text(i, prec+0.1, f'{prec:.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[0].set_ylabel('Precision (%)', fontweight='bold', fontsize=12)
axes[0].set_ylim([93, 100])
axes[0].grid(axis='y', alpha=0.4, linestyle='--')
axes[0].set_xticks(range(len(eval_df)))
axes[0].set_xticklabels(labels, fontsize=9.5)

# Recall
for i, (recall, label) in enumerate(zip(eval_df['Recall']*100, labels)):
    axes[1].bar(i, recall, color=colors[i], alpha=0.85, edgecolor='black', linewidth=1.5)
    axes[1].text(i, recall+0.5, f'{recall:.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[1].set_ylabel('Recall (%)', fontweight='bold', fontsize=12)
axes[1].set_ylim([65, 100])
axes[1].grid(axis='y', alpha=0.4, linestyle='--')
axes[1].set_xticks(range(len(eval_df)))
axes[1].set_xticklabels(labels, fontsize=9.5)

# F1 Score
for i, (f1, label) in enumerate(zip(eval_df['F1']*100, labels)):
    axes[2].bar(i, f1, color=colors[i], alpha=0.85, edgecolor='black', linewidth=1.5)
    axes[2].text(i, f1+0.2, f'{f1:.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[2].set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
axes[2].set_ylim([78, 96])
axes[2].grid(axis='y', alpha=0.4, linestyle='--')
axes[2].set_xticks(range(len(eval_df)))
axes[2].set_xticklabels(labels, fontsize=9.5)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: model_comparison.png")
plt.close()

# ============================================
# 3. ERROR METRICS: TP, FP, FN
# ============================================
print("🔧 Generating: Error Analysis...")

fig, ax = plt.subplots(figsize=(13, 6.5))

x = np.arange(len(eval_df))
width = 0.25

bars1 = ax.bar(x - width, eval_df['TP'], width, label='True Positives (TP)', 
              color='#2ECC71', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, eval_df['FP'], width, label='False Positives (FP)', 
              color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, eval_df['FN'], width, label='False Negatives (FN)', 
              color='#F39C12', alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_xlabel('Detection Strategy', fontweight='bold', fontsize=12)
ax.set_ylabel('Detection Count', fontweight='bold', fontsize=12)
ax.set_title('Detailed Error Analysis: True Positives vs False Positives vs False Negatives', 
            fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(axis='y', alpha=0.4, linestyle='--')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "error_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: error_analysis.png")
plt.close()

# ============================================
# 4. PRECISION-RECALL TRADEOFF (Scatter)
# ============================================
print("🔧 Generating: Precision-Recall Scatter Plot...")

fig, ax = plt.subplots(figsize=(11, 7.5))

scatter = ax.scatter(eval_df['Recall']*100, eval_df['Precision']*100, 
                    s=600, alpha=0.75, c=range(len(eval_df)), cmap='tab10', 
                    edgecolor='black', linewidth=2.5, zorder=3)

simple_labels = ['YOLO', 'YOLO+RF', 'YOLO+LR', 'Unanimous', 'Majority', 'Soft']
for i, (recall, prec, label) in enumerate(zip(eval_df['Recall']*100, 
                                              eval_df['Precision']*100, 
                                              simple_labels)):
    ax.annotate(label, (recall, prec), 
               xytext=(12, 12), textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                        edgecolor='black', alpha=0.8),
               ha='left', zorder=4)

ax.set_xlabel('Recall (%)', fontweight='bold', fontsize=13)
ax.set_ylabel('Precision (%)', fontweight='bold', fontsize=13)
ax.set_title('Precision-Recall Trade-off Analysis', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax.set_xlim([64, 100])
ax.set_ylim([92, 100.5])

plt.tight_layout()
plt.savefig(PLOTS_DIR / "precision_recall_scatter.png", dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: precision_recall_scatter.png")
plt.close()

# ============================================
# 5. FEATURE DISTRIBUTIONS
# ============================================
try:
    print("🔧 Generating: Feature Distributions...")
    
    tabular_df = pd.read_csv(OUTPUTS_DIR / "landmine_tabular_dataV3.csv")
    
    top_features = ['area', 'circularity', 'thermal_contrast', 
                   'intensity_std', 'thermal_gradient', 'max_min_ratio']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distributions: Mines (Red) vs Background (Blue)', 
                fontsize=13, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        mines = tabular_df[tabular_df['label'] == 1][feature].dropna()
        bg = tabular_df[tabular_df['label'] == 0][feature].dropna()
        
        ax.hist(mines, bins=25, alpha=0.65, color='#E74C3C', label='Mines', 
               edgecolor='black', linewidth=0.8)
        if len(bg) > 0:
            ax.hist(bg, bins=25, alpha=0.65, color='#3498DB', label='Background', 
                   edgecolor='black', linewidth=0.8)
        
        feature_title = feature.replace('_', ' ').title()
        ax.set_xlabel(feature_title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_distributions_all.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✅ Saved: feature_distributions_all.png")
    plt.close()
    
except Exception as e:
    print(f"   ⚠️  Skipped features: {str(e)[:40]}")

# ============================================
# 6. F1 SCORE RANKING
# ============================================
print("🔧 Generating: F1 Score Ranking...")

fig, ax = plt.subplots(figsize=(10, 6))

sorted_idx = np.argsort(eval_df['F1'].values)
sorted_labels = [labels[i] for i in sorted_idx]
sorted_f1 = eval_df['F1'].values[sorted_idx] * 100
sorted_colors = [colors[i] for i in sorted_idx]

bars = ax.barh(sorted_labels, sorted_f1, color=sorted_colors, alpha=0.85, 
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('F1 Score (%)', fontweight='bold', fontsize=12)
ax.set_title('Overall F1 Score Ranking by Strategy', fontweight='bold', fontsize=13)
ax.set_xlim([75, 96])

for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
    ax.text(score - 1, bar.get_y() + bar.get_height()/2, f'{score:.2f}%', 
           va='center', ha='right', fontweight='bold', fontsize=11, color='white')

ax.grid(axis='x', alpha=0.4, linestyle='--')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "f1_ranking.png", dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: f1_ranking.png")
plt.close()

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✨ VISUALIZATION GENERATION COMPLETE!")
print("=" * 60)
print(f"\n📁 Output Directory: {PLOTS_DIR}")
print("\n📊 Generated Visualizations:")
print("   • model_comparison.png")
print("   • error_analysis.png")
print("   • precision_recall_scatter.png")
print("   • feature_distributions_all.png")
print("   • f1_ranking.png")
print("\n✅ All charts ready for README.md!")
print("=" * 60 + "\n")
