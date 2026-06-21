#!/usr/bin/env python3
"""
Plot radial function histograms (physical and zernike distances per ring)
from the summary CSV with error bars showing uncertainties.
Automatically detects number of rows and creates subplots for each metric type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from pathlib import Path
import argparse
import re
import sys


def _ordered_summary_types(values):
    preferred_order = ['weighted', 'normal']
    summary_types = [summary_type for summary_type in preferred_order if summary_type in values]
    for summary_type in values:
        if summary_type not in summary_types:
            summary_types.append(summary_type)
    return summary_types


def _ring_ids_from_columns(columns, prefix):
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)_(?:mean|uncertainty)$')
    ring_ids = set()
    for column in columns:
        match = pattern.match(column)
        if match is not None:
            ring_ids.add(int(match.group(1)))
    return sorted(ring_ids)


def _mean_and_sample_uncertainty(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float('nan'), float('nan')
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return mean_value, 0.0
    uncertainty = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean_value, uncertainty


def _propagated_mean_uncertainty(uncertainties):
    uncertainties = np.asarray(uncertainties, dtype=float)
    uncertainties = uncertainties[np.isfinite(uncertainties)]
    if len(uncertainties) == 0:
        return float('nan')
    if len(uncertainties) == 1:
        return float(uncertainties[0])
    return float(np.sqrt(np.sum(uncertainties ** 2)) / len(uncertainties))


def _combined_uncertainty(sample_uncertainty, propagated_uncertainty):
    if not np.isfinite(sample_uncertainty) and not np.isfinite(propagated_uncertainty):
        return float('nan')
    if not np.isfinite(sample_uncertainty):
        return float(propagated_uncertainty)
    if not np.isfinite(propagated_uncertainty):
        return float(sample_uncertainty)
    return float(np.sqrt(sample_uncertainty ** 2 + propagated_uncertainty ** 2))


def _normalize_complex_series(values, uncertainties, mode):
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    normalized_values = values.copy()
    normalized_uncertainties = uncertainties.copy()

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0 or mode is None:
        return normalized_values, normalized_uncertainties

    if mode == 'max':
        scale = float(np.max(finite_values))
        if scale <= 0:
            normalized_values[np.isfinite(normalized_values)] = 0.0
            normalized_uncertainties[np.isfinite(normalized_uncertainties)] = 0.0
            return normalized_values, normalized_uncertainties
        normalized_values = normalized_values / scale
        normalized_uncertainties = normalized_uncertainties / scale
        return normalized_values, normalized_uncertainties

    if mode == 'minmax':
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        scale = max_value - min_value
        if scale <= 0:
            normalized_values[np.isfinite(normalized_values)] = 0.0
            normalized_uncertainties[np.isfinite(normalized_uncertainties)] = 0.0
            return normalized_values, normalized_uncertainties
        normalized_values = (normalized_values - min_value) / scale
        normalized_uncertainties = normalized_uncertainties / scale
        return normalized_values, normalized_uncertainties

    raise ValueError(f'Unsupported normalization mode: {mode}')


def _aggregate_metric_by_summary_type(df_summary, summary_types, ring_ids, metric_prefix, normalization_mode=None):
    aggregated = {}
    for summary_type in summary_types:
        group = df_summary[df_summary['summary_type'] == summary_type]
        ring_values = {}
        per_ring_values = {rid: [] for rid in ring_ids}
        per_ring_uncertainties = {rid: [] for rid in ring_ids}

        if 'complex_name' not in group.columns:
            raise ValueError('complex_name column is required to aggregate by complex')

        for _, complex_group in group.groupby('complex_name', sort=False):
            complex_means = []
            complex_uncertainties = []
            complex_valid_ring_ids = []

            for rid in ring_ids:
                mean_col = f'{metric_prefix}_ring{rid}_mean'
                unc_col = f'{metric_prefix}_ring{rid}_uncertainty'
                if mean_col not in complex_group.columns:
                    complex_means.append(float('nan'))
                    complex_uncertainties.append(float('nan'))
                    continue

                value = float(complex_group[mean_col].iloc[0])
                uncertainty = float(complex_group[unc_col].iloc[0]) if unc_col in complex_group.columns else float('nan')
                complex_means.append(value)
                complex_uncertainties.append(uncertainty)
                complex_valid_ring_ids.append(rid)

            normalized_means, normalized_uncertainties = _normalize_complex_series(
                np.array(complex_means, dtype=float),
                np.array(complex_uncertainties, dtype=float),
                normalization_mode,
            )

            for idx, rid in enumerate(ring_ids):
                if np.isfinite(normalized_means[idx]):
                    per_ring_values[rid].append(normalized_means[idx])
                if np.isfinite(normalized_uncertainties[idx]):
                    per_ring_uncertainties[rid].append(normalized_uncertainties[idx])

        for rid in ring_ids:
            finite_means = np.array(per_ring_values[rid], dtype=float)
            finite_uncertainties = np.array(per_ring_uncertainties[rid], dtype=float)
            if len(finite_means) == 0:
                ring_values[rid] = {
                    'mean': float('nan'),
                    'sample_uncertainty': float('nan'),
                    'propagated_uncertainty': float('nan'),
                    'combined_uncertainty': float('nan'),
                    'count': 0,
                }
                continue

            mean_value, sample_uncertainty = _mean_and_sample_uncertainty(finite_means)

            propagated_uncertainty = _propagated_mean_uncertainty(finite_uncertainties)

            combined_uncertainty = _combined_uncertainty(sample_uncertainty, propagated_uncertainty)

            ring_values[rid] = {
                'mean': mean_value,
                'sample_uncertainty': sample_uncertainty,
                'propagated_uncertainty': propagated_uncertainty,
                'combined_uncertainty': combined_uncertainty,
                'count': int(len(finite_means)),
            }
        aggregated[summary_type] = ring_values
    return aggregated


def _resolve_output_path(summary_path, output_path=None, output_name=None):
    input_stem = summary_path.stem.replace('_summary', '')
    default_stem = output_name or f'{input_stem}_mean_rdf'
    if output_path is None:
        return summary_path.parent / f'{default_stem}.pdf'

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / f'{default_stem}.pdf'
    if output_path.suffix:
        return output_path.with_name(f'{default_stem}{output_path.suffix}') if output_name else output_path
    return output_path / f'{default_stem}.pdf'


def _plot_metric_panel(ax, metric_data, ring_ids, summary_types, colors, markers, metric_label, value_label, show_legend=False):
    x_positions = np.arange(len(ring_ids))
    offset_step = 0.18 / max(len(summary_types), 1)
    uncertainty_scale = 3.0

    for type_idx, summary_type in enumerate(summary_types):
        ring_values = metric_data[summary_type]
        means = np.array([ring_values[rid]['mean'] for rid in ring_ids], dtype=float)
        sample_uncertainties = np.array([ring_values[rid]['sample_uncertainty'] for rid in ring_ids], dtype=float)
        combined_uncertainties = np.array([ring_values[rid]['combined_uncertainty'] for rid in ring_ids], dtype=float)

        x_offset = (type_idx - len(summary_types) / 2 + 0.5) * offset_step
        x_pos = x_positions + x_offset
        color = colors.get(summary_type, '#444444')
        ax.errorbar(
            x_pos,
            means,
            yerr=combined_uncertainties * uncertainty_scale,
            fmt='none',
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            capthick=1.4,
            zorder=2,
        )
        ax.scatter(
            x_pos,
            means,
            s=18,
            marker=markers.get(summary_type, 'o'),
            color=color,
            edgecolors='black',
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlabel('Ring ID', fontsize=12, fontweight='bold')
    ax.set_ylabel(value_label, fontsize=12, fontweight='bold')
    ax.set_title(metric_label, fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{r}' for r in ring_ids])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if show_legend:
        summary_handles = [
            Line2D(
                [0], [0],
                marker=markers.get(summary_type, 'o'),
                linestyle='none',
                markerfacecolor=colors.get(summary_type, '#444444'),
                markeredgecolor='black',
                markersize=8,
                label=summary_type,
            )
            for summary_type in summary_types
        ]
        legend1 = ax.legend(handles=summary_handles, fontsize=10, loc='best')
        ax.add_artist(legend1)


def plot_radial_histograms(summary_csv_path, output_path=None, output_name=None):
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
    ring_ids = _ring_ids_from_columns(df_summary.columns, 'physical_ring')
    print(f"  Number of rings: {len(ring_ids)}")
    
    summary_types = _ordered_summary_types(df_summary['summary_type'].dropna().unique().tolist())
    print(f"  Aggregating over {len(df_summary)} rows and {df_summary['complex_name'].nunique() if 'complex_name' in df_summary.columns else 'unknown'} complexes")

    plot_modes = [
        ('raw', None, 'Raw ring values'),
        ('max_norm', 'max', 'Per-complex max-normalized'),
        ('minmax_norm', 'minmax', 'Per-complex min-max normalized'),
    ]

    panel_data = {}
    for mode_name, normalization_mode, mode_label in plot_modes:
        physical_data = _aggregate_metric_by_summary_type(df_summary, summary_types, ring_ids, 'physical', normalization_mode=normalization_mode)
        zernike_data = _aggregate_metric_by_summary_type(df_summary, summary_types, ring_ids, 'zernike', normalization_mode=normalization_mode)
        panel_data[mode_name] = {
            'label': mode_label,
            'physical': physical_data,
            'zernike': zernike_data,
        }
        for summary_type in summary_types:
            counts = [physical_data[summary_type][rid]['count'] for rid in ring_ids]
            print(f"  {mode_name} / {summary_type}: ring sample counts {counts}")

    # Create figure with three rows and two columns
    fig, axes = plt.subplots(len(plot_modes), 2, figsize=(16, 18), sharex=True)
    
    # Define colors and markers for each summary type
    colors = {
        'weighted': '#1f77b4',
        'normal': '#ff7f0e',
    }
    markers = {
        'weighted': 'o',
        'normal': 's',
    }
    
    for row_index, (mode_name, _, mode_label) in enumerate(plot_modes):
        show_legend = (row_index == 0) and ('weighted' in summary_types and 'normal' in summary_types)
        _plot_metric_panel(
            axes[row_index, 0],
            panel_data[mode_name]['physical'],
            ring_ids,
            summary_types,
            colors,
            markers,
            f'{mode_label} - Physical Distance per Ring',
            'Physical Distance (Å)',
            show_legend=show_legend,
        )
        _plot_metric_panel(
            axes[row_index, 1],
            panel_data[mode_name]['zernike'],
            ring_ids,
            summary_types,
            colors,
            markers,
            f'{mode_label} - Zernike Distance per Ring',
            'Zernike Distance',
            show_legend=False,
        )

        if row_index == 0:
            axes[row_index, 0].set_ylabel('Physical Distance (Å)', fontsize=12, fontweight='bold')
            axes[row_index, 1].set_ylabel('Zernike Distance', fontsize=12, fontweight='bold')

        row_label = mode_label
        axes[row_index, 0].text(
            -0.18,
            0.5,
            row_label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=12,
            fontweight='bold',
        )

    fig.suptitle(f'Radial Functions with Uncertainties ({summary_path.stem})',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    # Save output
    output_path = _resolve_output_path(summary_path, output_path, output_name)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot radial function histograms with uncertainties from summary CSV'
    )
    parser.add_argument('summary_csv', help='Path to summary.csv from get_complementary_plane2.py')
    parser.add_argument('-o', '--output', help='Output path for plot (default: <stem>_radial_histograms.png)')
    parser.add_argument('--output-name', help='Output file name without extension (default: <summary>_radial_histograms)')
    
    args = parser.parse_args()
    
    try:
        plot_radial_histograms(args.summary_csv, args.output, args.output_name)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
