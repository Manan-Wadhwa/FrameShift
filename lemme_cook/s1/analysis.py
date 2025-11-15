#!/usr/bin/env python
#
# CELL 7.9: ANALYZE RESULTS
# CELL 7.10: VISUALIZE PERFORMANCE
# CELL 7.11: FINAL SUMMARY & RECOMMENDATIONS
#
# This script reads all worker .json files from a directory,
# aggregates them, and runs the full analysis.
#

print("--- [Starting] Analysis Script ---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
import sys

# Define output directory for analysis results
ANALYSIS_OUTPUT_DIR = "analysis_output"

#%%
#
# --- 1. Load All Worker Results ---
#
def load_results(input_dir):
    print(f"Loading all results from: {input_dir}")
    results_path = Path(input_dir)
    all_json_files = list(results_path.glob("results_*.json"))
    
    if not all_json_files:
        print(f"Error: No 'results_*.json' files found in {input_dir}")
        print("Please run the benchmark workers first.")
        sys.exit(1)
        
    benchmark_results = []
    for f in all_json_files:
        with open(f, 'r') as fin:
            data = json.load(fin)
            benchmark_results.extend(data)
            
    print(f"Loaded {len(benchmark_results)} results from {len(all_json_files)} worker files.")
    
    df = pd.DataFrame(benchmark_results)
    return df

#%%
#
# CELL 7.9: ANALYZE RESULTS
#
def run_analysis(df):
    print("\n--- [CELL 7.9] Results Analysis ---")
    
    # Filter to only results with ground truth
    df_with_gt = df[df['has_ground_truth'] == True].copy()
    
    # Ensure numeric types after loading from JSON
    for col in ['f1', 'iou', 'precision', 'recall', 'time']:
        df_with_gt[col] = pd.to_numeric(df_with_gt[col], errors='coerce')
    
    # Drop rows where metrics are NaN (e.g., failed runs)
    df_with_gt = df_with_gt.dropna(subset=['f1', 'time'])
    
    print(f"\nResults Summary:")
    print(f"  Total tests: {len(df)}")
    print(f"  With ground truth: {len(df_with_gt)}")
    print(f"  Failed/Skipped tests (NaN F1): {df['f1'].isna().sum()}")

    if len(df_with_gt) == 0:
        print("\n! No results with ground truth available for analysis!")
        return None, None
        
    analysis_results = {}

    # ANALYSIS 1: Overall Rankings
    print("\n" + "="*80)
    print("  OVERALL RANKINGS (Average across all test cases)")
    print("="*80)
    overall = df_with_gt.groupby('detector').agg({
        'f1': 'mean', 'iou': 'mean', 'precision': 'mean', 'recall': 'mean', 'time': 'mean'
    }).round(4)
    overall = overall.sort_values('f1', ascending=False)
    overall['time_ms'] = (overall['time'] * 1000).astype(int)
    print("\n", overall[['f1', 'iou', 'precision', 'recall', 'time_ms']])
    analysis_results['overall'] = overall

    # ANALYSIS 2: By Preprocessing Pipeline
    print("\n" + "="*80)
    print("  IMPACT OF PREPROCESSING")
    print("="*80)
    by_preprocessing = df_with_gt.groupby('preprocessing').agg({
        'f1': 'mean', 'iou': 'mean', 'time': 'mean'
    }).round(4)
    by_preprocessing = by_preprocessing.sort_values('f1', ascending=False)
    print("\n", by_preprocessing)
    winner_preprocessing = by_preprocessing.index[0]
    print(f"\n✨ Best Preprocessing: {winner_preprocessing}")
    if 'Raw' in by_preprocessing.index:
        improvement = (by_preprocessing.loc[winner_preprocessing, 'f1'] - by_preprocessing.loc['Raw', 'f1']) * 100
        print(f"  Improvement over Raw: +{improvement:.1f}% F1")
    analysis_results['by_preprocessing'] = by_preprocessing
    analysis_results['winner_preprocessing'] = winner_preprocessing

    # ANALYSIS 3: By Defect Type
    print("\n" + "="*80)
    print("  PERFORMANCE BY DEFECT TYPE")
    print("="*80)
    by_defect = df_with_gt.groupby('defect_type').agg({
        'f1': 'mean', 'iou': 'mean'
    }).round(4)
    by_defect = by_defect.sort_values('f1', ascending=False)
    print("\n", by_defect)
    analysis_results['by_defect'] = by_defect

    # ANALYSIS 4: Speed vs Accuracy Tradeoff
    print("\n" + "="*80)
    print("  SPEED VS ACCURACY TRADEOFF")
    print("="*80)
    tradeoff = overall[['f1', 'time_ms']].copy().sort_values('time_ms')
    print("\nFastest Methods (< 100ms):")
    fast_methods = tradeoff[tradeoff['time_ms'] < 100]
    print(fast_methods)
    print("\nMost Accurate Methods (F1 > 0.5):")
    accurate_methods = tradeoff[tradeoff['f1'] > 0.5].sort_values('f1', ascending=False)
    print(accurate_methods)
    analysis_results['tradeoff'] = tradeoff
    
    # ANALYSIS 5: Best Combination per Defect Type
    print("\n" + "="*80)
    print("  RECOMMENDED COMBINATIONS BY DEFECT TYPE")
    print("="*80)
    recommendations = {}
    recommendations['by_defect_type'] = {}
    for defect_type in df_with_gt['defect_type'].unique():
        df_defect = df_with_gt[df_with_gt['defect_type'] == defect_type]
        best_combo = df_defect.loc[df_defect['f1'].idxmax()]
        print(f"\n{defect_type.upper()}:")
        print(f"  Best: {best_combo['preprocessing']} + {best_combo['detector']}")
        print(f"  F1={best_combo['f1']:.3f}, IoU={best_combo['iou']:.3f}, {best_combo['time']*1000:.0f}ms")
        recommendations['by_defect_type'][defect_type] = {
            'preprocessing': best_combo['preprocessing'],
            'detector': best_combo['detector'],
            'f1': float(best_combo['f1']),
            'iou': float(best_combo['iou']),
            'time_ms': int(best_combo['time'] * 1000)
        }
        
    recommendations['fastest'] = {
        'name': tradeoff.index[0],
        'f1': float(tradeoff.iloc[0]['f1']),
        'time_ms': int(tradeoff.iloc[0]['time_ms'])
    }
    recommendations['most_accurate'] = {
        'name': overall.index[0],
        'f1': float(overall.iloc[0]['f1']),
        'time_ms': int(overall.iloc[0]['time_ms'])
    }
    recommendations['best_preprocessing'] = winner_preprocessing
    
    analysis_results['recommendations'] = recommendations
    
    return df_with_gt, analysis_results

#%%
#
# CELL 7.10: VISUALIZE PERFORMANCE
#
def run_visualization(df_with_gt, analysis_results, output_dir):
    print("\n--- [CELL 7.10] Performance Visualization ---")
    
    if df_with_gt is None or analysis_results is None:
        print("! No data available for visualization, skipping.")
        return

    # --- Plot 1: F1 Score by Method ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1 = axes[0, 0]
    method_f1 = analysis_results['overall']['f1'].sort_values(ascending=True)
    method_f1.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('F1 Score')
    ax1.set_title('Average F1 Score by Detection Method', fontweight='bold')
    
    # --- Plot 2: Speed vs Accuracy Scatter ---
    ax2 = axes[0, 1]
    tradeoff = analysis_results['tradeoff']
    ax2.scatter(tradeoff['time_ms'], tradeoff['f1'], s=100, alpha=0.6, c=range(len(tradeoff)), cmap='viridis')
    for idx, row in tradeoff.iterrows():
        ax2.annotate(idx, (row['time_ms'], row['f1']), fontsize=8)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Speed vs Accuracy Tradeoff', fontweight='bold')
    
    # --- Plot 3: Preprocessing Impact ---
    ax3 = axes[1, 0]
    prep_f1 = analysis_results['by_preprocessing']['f1'].sort_values(ascending=True)
    prep_f1.plot(kind='barh', ax=ax3, color='coral')
    ax3.set_xlabel('F1 Score')
    ax3.set_title('Impact of Preprocessing on F1 Score', fontweight='bold')
    
    # --- Plot 4: Performance by Defect Type ---
    ax4 = axes[1, 1]
    defect_perf = df_with_gt.groupby(['defect_type', 'detector'])['f1'].mean().unstack()
    top_methods = analysis_results['overall'].index[:5]
    defect_perf[top_methods].plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_xlabel('Defect Type')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Performance by Defect Type (Top 5 Methods)', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    viz_path = output_dir / 'performance_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to: {viz_path}")
    
    # --- Create comparison matrix heatmap ---
    print("Generating comparison heatmap...")
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot = df_with_gt.pivot_table(index='detector', columns='preprocessing', values='f1', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'F1 Score'})
    ax.set_title('F1 Scores: Detection Method vs. Preprocessing Pipeline', fontweight='bold', fontsize=14)
    plt.tight_layout()
    heatmap_path = output_dir / 'method_preprocessing_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"  Heatmap saved to: {heatmap_path}")

#%%
#
# CELL 7.11: FINAL SUMMARY & HACKATHON RECOMMENDATIONS
#
def print_recommendations(analysis_results, output_dir):
    print("\n--- [CELL 7.11] Final Summary & Hackathon Recommendations ---")
    
    if analysis_results is None:
        print("! No recommendations available, analysis failed.")
        return

    recs = analysis_results['recommendations']
    
    print("\n" + "="*80)
    print("  HACKATHON DEMO RECOMMENDATIONS")
    print("="*80)
    
    print("\n  FOR REAL-TIME DEMO (SPEED):")
    fastest = recs['fastest']
    print(f"    Method: {fastest['name']}")
    print(f"    Performance: F1={fastest['f1']:.3f}")
    print(f"    Speed: {fastest['time_ms']}ms")
    
    print("\n  FOR ACCURACY (QUALITY):")
    accurate = recs['most_accurate']
    print(f"    Method: {accurate['name']}")
    print(f"    Performance: F1={accurate['f1']:.3f}")
    print(f"    Speed: {accurate['time_ms']}ms")

    print("\n  BEST PREPROCESSING:")
    print(f"    Pipeline: {recs['best_preprocessing']}")
    
    print("\n  BY DEFECT TYPE:")
    for defect_type, combo in recs['by_defect_type'].items():
        print(f"\n    {defect_type.upper()}:")
        print(f"      {combo['preprocessing']} + {combo['detector']}")
        print(f"      F1={combo['f1']:.3f}, {combo['time_ms']}ms")
        
    print("\n" + "="*80)
    print("  QUICK COMPARISON TABLE (Copy this for your slides!)")
    print("="*80)
    print(f"{'Method':<25} | {'F1 Score':<10} | {'Speed (ms)':<10} | {'Use Case':<10}")
    print("-" * 65)
    top_methods = analysis_results['overall'].head(5)
    for idx, row in top_methods.iterrows():
        use_case = 'Fast' if row['time_ms'] < 100 else 'Accurate'
        print(f"{idx:<25} | {row['f1']:.3f}      | {int(row['time_ms']):<10} | {use_case}")
    
    # Save recommendations JSON
    rec_path = output_dir / 'recommendations.json'
    with open(rec_path, 'w') as f:
        json.dump(recs, f, indent=2)
    print(f"\n  Recommendations saved to: {rec_path}")
    
#%%
#
# --- Main Analysis Execution ---
#
def main(args):
    
    # Create output directory
    output_dir = Path(ANALYSIS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df_full = load_results(args.input_dir)
    # Save full CSV results
    csv_path = output_dir / 'benchmark_results.csv'
    df_full.to_csv(csv_path, index=False)
    print(f"  Full raw results saved to: {csv_path}")
    
    # 2. Run Analysis
    df_with_gt, analysis_results = run_analysis(df_full)
    
    # 3. Run Visualization
    run_visualization(df_with_gt, analysis_results, output_dir)
    
    # 4. Print Recommendations
    print_recommendations(analysis_results, output_dir)
    
    # 5. Save Summary Tables
    summary_path = output_dir / 'summary_tables.txt'
    with open(summary_path, 'w') as f:
        if analysis_results:
            f.write("="*80 + "\nOVERALL RANKINGS\n" + "="*80 + "\n")
            f.write(analysis_results['overall'].to_string())
            f.write("\n\n" + "="*80 + "\nBY PREPROCESSING\n" + "="*80 + "\n")
            f.write(analysis_results['by_preprocessing'].to_string())
            f.write("\n\n" + "="*80 + "\nBY DEFECT TYPE\n" + "="*80 + "\n")
            f.write(analysis_results['by_defect'].to_string())
            f.write("\n\n" + "="*80 + "\nSPEED VS ACCURACY\n" + "="*80 + "\n")
            f.write(analysis_results['tradeoff'].to_string())
        else:
            f.write("No ground truth data found for analysis.")
            
    print(f"  Summary tables saved to: {summary_path}")
    print("\n--- ✅ Analysis Complete ---")
    print(f"  All outputs saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full benchmark analysis.")
    parser.add_argument("-i", "--input-dir", default="benchmark_results", 
                        help="Directory to read worker .json files from")
    args = parser.parse_args()
    main(args)
