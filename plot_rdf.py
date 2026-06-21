#!/usr/bin/env python3
"""
Plot radial function histograms (physical and zernike distances per ring)
from the summary CSV with error bars showing uncertainties.
Automatically detects number of rows and creates subplots for each metric type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def _default_output_stem(summary_path):
    stem = summary_path.stem
    if stem.endswith('_summary'):
        return stem[:-len('_summary')]
    return stem


def plot_radial_histograms(summary_csv_path, output_path=None):
    """
    Create two subplots showing ring-by-ring histograms with uncertainties
    for physical and zernike distances.
    
    Parameters
    ----------
    summary_csv_path : str or Path
        Path to the summary CSV file from get_complementary_plane2.py
    output_path : str or Path, optional
        Output path for the plot. If None, uses same stem as input with .png
    """
    
    summary_path = Path(summary_csv_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    # Read summary CSV
    df_summary = pd.read_csv(summary_path)
    print(f"✓ Loaded summary from {summary_path}")
    print(f"  Shape: {df_summary.shape}")
    print(f"  Summary types: {df_summary['summary_type'].unique().tolist()}")
    
    # Extract ring IDs from columns
    ring_cols_physical = [c for c in df_summary.columns if c.startswith('physical_ring')]
    ring_ids = sorted(set(
        int(c.replace('physical_ring', '').replace('_mean', '').replace('_uncertainty', ''))
        for c in ring_cols_physical
    ))
    print(f"  Number of rings: {len(ring_ids)}")
    
    # Prepare data: extract means and uncertainties for each ring and summary type
    physical_data = {}
    zernike_data = {}
    
    for idx, row in df_summary.iterrows():
        summary_type = row['summary_type']
        
        physical_means = []
        physical_uncs = []
        zernike_means = []
        zernike_uncs = []
        
        for rid in ring_ids:
            phys_mean = row.get(f'physical_ring{rid}_mean', np.nan)
            phys_unc = row.get(f'physical_ring{rid}_uncertainty', np.nan)
            zern_mean = row.get(f'zernike_ring{rid}_mean', np.nan)
            zern_unc = row.get(f'zernike_ring{rid}_uncertainty', np.nan)
            
            physical_means.append(phys_mean)
            physical_uncs.append(phys_unc)
            zernike_means.append(zern_mean)
            zernike_uncs.append(zern_unc)
        
        physical_data[summary_type] = (np.array(physical_means), np.array(physical_uncs))
        zernike_data[summary_type] = (np.array(zernike_means), np.array(zernike_uncs))
    
    # Create figure with two subplots (one for physical, one for zernike)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors and markers for each summary type
    colors = {'weighted': '#1f77b4', 'normal': '#ff7f0e'}
    markers = {'weighted': 'o', 'normal': 's'}
    
    # Offset for multiple points per ring
    summary_types = sorted(physical_data.keys())
    n_summary_types = len(summary_types)
    offset_step = 0.15 / n_summary_types
    has_both_types = 'weighted' in summary_types and 'normal' in summary_types
    
    # Plot 1: Physical distance per ring
    ax = axes[0]
    for type_idx, summary_type in enumerate(summary_types):
        means, uncs = physical_data[summary_type]
        x_offset = (type_idx - n_summary_types/2 + 0.5) * offset_step
        x_pos = np.arange(len(ring_ids)) + x_offset
        
        ax.errorbar(x_pos, means, yerr=uncs, fmt='none', 
                   color=colors[summary_type], elinewidth=1.5, capsize=4, capthick=1.5,
                   alpha=0.7, zorder=2)
        ax.scatter(x_pos, means, s=80, marker=markers[summary_type], 
                  color=colors[summary_type], label=summary_type, 
                  edgecolors='black', linewidth=0.8, alpha=0.9, zorder=3)
    
    ax.set_xlabel('Ring ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Physical Distance (Å)', fontsize=12, fontweight='bold')
    ax.set_title('Physical Distance per Ring', fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(len(ring_ids)))
    ax.set_xticklabels([f'{r}' for r in ring_ids])
    if has_both_types:
        ax.legend(title='Summary Type', fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Zernike distance per ring
    ax = axes[1]
    for type_idx, summary_type in enumerate(summary_types):
        means, uncs = zernike_data[summary_type]
        x_offset = (type_idx - n_summary_types/2 + 0.5) * offset_step
        x_pos = np.arange(len(ring_ids)) + x_offset
        
        ax.errorbar(x_pos, means, yerr=uncs, fmt='none', 
                   color=colors[summary_type], elinewidth=1.5, capsize=4, capthick=1.5,
                   alpha=0.7, zorder=2)
        ax.scatter(x_pos, means, s=80, marker=markers[summary_type], 
                  color=colors[summary_type], label=summary_type, 
                  edgecolors='black', linewidth=0.8, alpha=0.9, zorder=3)
    
    ax.set_xlabel('Ring ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Zernike Distance', fontsize=12, fontweight='bold')
    ax.set_title('Zernike Distance per Ring', fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(len(ring_ids)))
    ax.set_xticklabels([f'{r}' for r in ring_ids])
    if has_both_types:
        ax.legend(title='Summary Type', fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle(f'Radial Functions with Uncertainties ({summary_path.stem})', 
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    # Save output
    if output_path is None:
        output_path = summary_path.parent / f'{_default_output_stem(summary_path)}_rdf.pdf'
    else:
        output_path = Path(output_path)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot radial function histograms with uncertainties from summary CSV'
    )
    parser.add_argument('summary_csv', help='Path to summary.csv from get_complementary_plane2.py')
    parser.add_argument('-o', '--output', help='Output path for plot (default: <stem without _summary>_rdf.png)')
    
    args = parser.parse_args()
    
    try:
        plot_radial_histograms(args.summary_csv, args.output)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
