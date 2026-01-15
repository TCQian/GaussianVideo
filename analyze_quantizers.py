#!/usr/bin/env python3
"""
Simple script to analyze quantizer usage and similarity.
Uses the built-in analyze() methods in each quantizer.
"""

import torch
import numpy as np
from quantize import UniformQuantizer, VectorQuantizer
import matplotlib.pyplot as plt


def analyze_quantizer_similarity(model, layer=1):
    """
    Analyze similarity in quantized data using built-in quantizer methods.
    
    Args:
        model: GaussianVideo3D2D model instance
        layer: Layer to analyze (0 or 1)
    """
    print(f"\n{'='*80}")
    print(f"QUANTIZER SIMILARITY ANALYSIS - Layer {layer}")
    print(f"{'='*80}\n")
    
    # Get data based on layer
    if layer == 0:
        cholesky_data = model._cholesky_3D
        features_data = model._features_dc_3D
        cholesky_quantizer = model.cholesky_quantizer_layer0
        features_quantizer = model.features_dc_quantizer_layer0
    else:
        cholesky_data = model._cholesky_2D
        features_data = model._features_dc_2D
        cholesky_quantizer = model.cholesky_quantizer_layer1
        features_quantizer = model.features_dc_quantizer_layer1
    
    # Analyze Cholesky quantizer
    print("\n" + "="*80)
    print("CHOLESKY QUANTIZER ANALYSIS")
    print("="*80)
    cholesky_results = cholesky_quantizer.analyze(cholesky_data, verbose=True)
    
    # Analyze Features quantizer
    print("\n" + "="*80)
    print("FEATURES (COLOR) QUANTIZER ANALYSIS")
    print("="*80)
    features_results = features_quantizer.analyze(features_data, verbose=True)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    # Cholesky recommendations
    cholesky_util = cholesky_results['utilization']
    if cholesky_util < 0.3:
        print(f"\n✓ Cholesky: High similarity detected ({cholesky_util*100:.1f}% utilization)")
        print(f"  → Significant compression benefit from entropy coding")
        print(f"  → Compression ratio: {cholesky_results['compression_ratio']:.2f}x")
        print(f"  → Potential bit reduction: {cholesky_results['current_bits'] - cholesky_results['entropy_coded_bits']:,.0f} bits")
    elif cholesky_util < 0.6:
        print(f"\n○ Cholesky: Moderate similarity ({cholesky_util*100:.1f}% utilization)")
        print(f"  → Some compression benefit possible")
        print(f"  → Compression ratio: {cholesky_results['compression_ratio']:.2f}x")
    else:
        print(f"\n✗ Cholesky: Low similarity ({cholesky_util*100:.1f}% utilization)")
        print(f"  → Limited compression benefit from similarity")
    
    # Features recommendations
    features_util = features_results['utilization']
    optimal_codebook = max([s['unique_count'] for s in features_results['per_quantizer_stats']])
    current_codebook = features_results['codebook_size']
    
    if features_util < 0.3:
        print(f"\n✓ Features: High similarity detected ({features_util*100:.1f}% utilization)")
        print(f"  → Only {optimal_codebook} unique indices used out of {current_codebook} codebook entries")
        print(f"  → Consider reducing codebook size to {optimal_codebook * 2} (safety margin)")
        print(f"  → Compression ratio: {features_results['compression_ratio']:.2f}x")
        potential_bits = features_results['current_bits'] * (1 - optimal_codebook / current_codebook)
        print(f"  → Potential bit reduction from smaller codebook: ~{potential_bits:,.0f} bits")
    elif features_util < 0.6:
        print(f"\n○ Features: Moderate similarity ({features_util*100:.1f}% utilization)")
        print(f"  → {optimal_codebook} unique indices used out of {current_codebook} codebook entries")
        print(f"  → Some benefit from smaller codebook possible")
    else:
        print(f"\n✗ Features: Low similarity ({features_util*100:.1f}% utilization)")
        print(f"  → Most codebook entries are used")
        print(f"  → Limited benefit from codebook reduction")
    
    return {
        'cholesky': cholesky_results,
        'features': features_results
    }


def plot_quantizer_distributions(results, output_path=None):
    """Create simple visualizations of quantizer distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cholesky value distribution
    cholesky_dist = results['cholesky']['value_distribution']
    values = list(cholesky_dist.keys())
    counts = list(cholesky_dist.values())
    axes[0, 0].bar(values, counts, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Cholesky: Value Distribution')
    axes[0, 0].set_xlabel('Quantized Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cholesky top values
    top_cholesky = results['cholesky']['most_common'][:10]
    top_vals = [x[0] for x in top_cholesky]
    top_counts = [x[1] for x in top_cholesky]
    axes[0, 1].barh(range(len(top_vals)), top_counts, alpha=0.7, edgecolor='black')
    axes[0, 1].set_yticks(range(len(top_vals)))
    axes[0, 1].set_yticklabels([f"Val {v}" for v in top_vals])
    axes[0, 1].set_title('Cholesky: Top 10 Most Common Values')
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Features per-quantizer usage
    features_stats = results['features']['per_quantizer_stats']
    for q, stats in enumerate(features_stats):
        dist = stats['distribution']
        values = list(dist.keys())
        counts = list(dist.values())
        axes[1, 0].bar([v + q*0.3 for v in values], counts, alpha=0.7, 
                      width=0.3, label=f'Quantizer {q}', edgecolor='black')
    axes[1, 0].set_title('Features: Codebook Usage per Quantizer')
    axes[1, 0].set_xlabel('Codebook Index')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Features top index combinations
    top_features = results['features']['most_common_indices'][:10]
    top_combos = [str(x[0])[:20] for x in top_features]  # Truncate long strings
    top_counts = [x[1] for x in top_features]
    axes[1, 1].barh(range(len(top_combos)), top_counts, alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_combos)))
    axes[1, 1].set_yticklabels(top_combos, fontsize=8)
    axes[1, 1].set_title('Features: Top 10 Index Combinations')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Example usage
    print("Quantizer Analysis Tool")
    print("="*80)
    print("\nThis tool uses the built-in analyze() methods in each quantizer.")
    print("To use with your model:")
    print("  1. Load your model")
    print("  2. Call: analyze_quantizer_similarity(model, layer=1)")
    print("  3. Optionally plot: plot_quantizer_distributions(results)")
    print("\nExample:")
    print("  from gaussianvideo3D2D import GaussianVideo3D2D")
    print("  from analyze_quantizers import analyze_quantizer_similarity")
    print("  ")
    print("  model = GaussianVideo3D2D(...)")
    print("  # Load checkpoint...")
    print("  results = analyze_quantizer_similarity(model, layer=1)")
