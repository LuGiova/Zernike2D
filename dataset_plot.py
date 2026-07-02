#!/usr/bin/env python3
"""
Plot radial function histograms (physical and zernike distances per ring)
from the summary CSV with error bars showing uncertainties.
Automatically detects number of rows and creates subplots for each metric type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def _aggregate_real_metric_by_summary_type(
    df_summary,
    summary_types,
    ring_ids,
    metric_prefix,
    normalization_mode=None,
    bin_width=2.0,
    max_x=50.0,
):
    aggregated = {}
    bin_edges = np.arange(0.0, max_x + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2.0
    bin_ids = list(range(len(bin_centers)))

    for summary_type in summary_types:
        group = df_summary[df_summary['summary_type'] == summary_type]
        bin_values = {}
        per_bin_values = {bid: [] for bid in bin_ids}
        per_bin_uncertainties = {bid: [] for bid in bin_ids}

        if 'complex_name' not in group.columns:
            raise ValueError('complex_name column is required to aggregate by complex')
        if 'radius' not in group.columns:
            raise ValueError('radius column is required to compute real-valued ring positions')

        for _, complex_group in group.groupby('complex_name', sort=False):
            radius_values = pd.to_numeric(complex_group['radius'], errors='coerce').to_numpy(dtype=float)
            radius_values = radius_values[np.isfinite(radius_values)]
            if len(radius_values) == 0:
                continue

            radius = float(radius_values[0])
            if not np.isfinite(radius) or radius <= 0:
                continue

            complex_means = []
            complex_uncertainties = []
            complex_x_positions = []

            for rid in ring_ids:
                mean_col = f'{metric_prefix}_ring{rid}_mean'
                unc_col = f'{metric_prefix}_ring{rid}_uncertainty'
                if mean_col not in complex_group.columns:
                    complex_means.append(float('nan'))
                    complex_uncertainties.append(float('nan'))
                    complex_x_positions.append(float('nan'))
                    continue

                value = float(complex_group[mean_col].iloc[0])
                uncertainty = float(complex_group[unc_col].iloc[0]) if unc_col in complex_group.columns else float('nan')
                x_position = (rid - 0.5) * radius / 10.0

                complex_means.append(value)
                complex_uncertainties.append(uncertainty)
                complex_x_positions.append(x_position)

            normalized_means, normalized_uncertainties = _normalize_complex_series(
                np.array(complex_means, dtype=float),
                np.array(complex_uncertainties, dtype=float),
                normalization_mode,
            )

            for idx, x_position in enumerate(complex_x_positions):
                if not np.isfinite(x_position) or x_position > max_x:
                    continue

                bin_index = int(np.floor(x_position / bin_width))
                bin_index = min(bin_index, len(bin_ids) - 1)

                if np.isfinite(normalized_means[idx]):
                    per_bin_values[bin_index].append(normalized_means[idx])
                if np.isfinite(normalized_uncertainties[idx]):
                    per_bin_uncertainties[bin_index].append(normalized_uncertainties[idx])

        for bid in bin_ids:
            finite_means = np.array(per_bin_values[bid], dtype=float)
            finite_uncertainties = np.array(per_bin_uncertainties[bid], dtype=float)
            if len(finite_means) == 0:
                bin_values[bid] = {
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

            bin_values[bid] = {
                'mean': mean_value,
                'sample_uncertainty': sample_uncertainty,
                'propagated_uncertainty': propagated_uncertainty,
                'combined_uncertainty': combined_uncertainty,
                'count': int(len(finite_means)),
            }

        aggregated[summary_type] = {
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'bin_values': bin_values,
        }
    return aggregated


def _resolve_output_path(summary_path, output_path=None, output_name=None, suffix=''):
    input_stem = summary_path.stem.replace('_summary', '')
    default_stem = output_name or f'{input_stem}_mean_rdf'
    default_stem = f'{default_stem}{suffix}'
    if output_path is None:
        return summary_path.parent / f'{default_stem}.pdf'

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / f'{default_stem}.pdf'
    if output_path.suffix:
        return output_path.with_name(f'{default_stem}{output_path.suffix}') if output_name or suffix else output_path
    return output_path / f'{default_stem}.pdf'


def _resolve_real_output_path(summary_path, output_path=None, output_name=None):
    input_stem = summary_path.stem.replace('_summary', '')
    default_stem = output_name or f'{input_stem}_real_rdf'
    if output_path is None:
        return summary_path.parent / f'{default_stem}.pdf'

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / f'{default_stem}.pdf'
    if output_path.suffix:
        return output_path.with_name(f'{default_stem}{output_path.suffix}') if output_name else output_path
    return output_path / f'{default_stem}.pdf'


def _display_name_from_summary_path(summary_path):
    name = Path(summary_path).name
    if name.endswith('_summary.csv'):
        return name[:-len('_summary.csv')]
    stem = Path(name).stem
    return stem[:-len('_summary')] if stem.endswith('_summary') else stem


def _plot_metric_panel(
    ax,
    metric_data_by_source,
    ring_ids,
    summary_types,
    colors,
    markers,
    metric_label,
    value_label,
    source_labels=None,
):
    x_positions = np.arange(len(ring_ids))
    source_count = len(metric_data_by_source)
    offset_step = 0.18 / max(len(summary_types) * max(source_count, 1), 1)
    uncertainty_scale = 3.0
    source_styles = [
        {'marker': 's', 'color': '#1f77b4'},
        {'marker': 'o', 'color': '#ff7f0e'},
    ]

    for source_idx, metric_data in enumerate(metric_data_by_source):
        source_style = source_styles[source_idx % len(source_styles)]
        for type_idx, summary_type in enumerate(summary_types):
            ring_values = metric_data[summary_type]
            means = np.array([ring_values[rid]['mean'] for rid in ring_ids], dtype=float)
            combined_uncertainties = np.array([ring_values[rid]['combined_uncertainty'] for rid in ring_ids], dtype=float)

            x_offset = ((source_idx * len(summary_types)) + type_idx - (source_count * len(summary_types)) / 2 + 0.5) * offset_step
            x_pos = x_positions + x_offset
            color = source_style['color']
            marker = source_style['marker']

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
                marker=marker,
                facecolors=color,
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

    if source_labels and len(source_labels) > 1:
        source_handles = []
        for source_index, source_label in enumerate(source_labels):
            marker = source_styles[source_index % len(source_styles)]['marker']
            color = source_styles[source_index % len(source_styles)]['color']
            source_handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle='none',
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8,
                    label=source_label,
                )
            )
        plt.sca(ax)
        plt.legend(handles=source_handles, fontsize=10, loc='best', title='file')


def _plot_real_metric_panel(
    ax,
    metric_data_by_source,
    bin_centers,
    summary_types,
    metric_label,
    value_label,
    source_labels=None,
):
    x_positions = np.asarray(bin_centers, dtype=float)
    source_count = len(metric_data_by_source)
    offset_step = 0.18 / max(len(summary_types) * max(source_count, 1), 1)
    uncertainty_scale = 3.0
    source_styles = [
        {'marker': 's', 'color': '#1f77b4'},
        {'marker': 'o', 'color': '#ff7f0e'},
    ]

    for source_idx, metric_data in enumerate(metric_data_by_source):
        source_style = source_styles[source_idx % len(source_styles)]
        for type_idx, summary_type in enumerate(summary_types):
            bin_values = metric_data[summary_type]['bin_values']
            means = np.array([bin_values[bid]['mean'] for bid in range(len(bin_centers))], dtype=float)
            combined_uncertainties = np.array([bin_values[bid]['combined_uncertainty'] for bid in range(len(bin_centers))], dtype=float)

            x_offset = ((source_idx * len(summary_types)) + type_idx - (source_count * len(summary_types)) / 2 + 0.5) * offset_step
            x_pos = x_positions + x_offset
            color = source_style['color']
            marker = source_style['marker']

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
                marker=marker,
                facecolors=color,
                edgecolors='black',
                linewidth=0.8,
                alpha=0.95,
                zorder=3,
            )

    ax.set_xlabel('Real radial distance bin center (Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel(value_label, fontsize=12, fontweight='bold')
    ax.set_title(metric_label, fontsize=13, fontweight='bold')
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f'{center:.0f}' for center in bin_centers], rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if source_labels and len(source_labels) > 1:
        source_handles = []
        for source_index, source_label in enumerate(source_labels):
            marker = source_styles[source_index % len(source_styles)]['marker']
            color = source_styles[source_index % len(source_styles)]['color']
            source_handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle='none',
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8,
                    label=source_label,
                )
            )
        plt.sca(ax)
        plt.legend(handles=source_handles, fontsize=10, loc='best', title='file')


def plot_radial_histograms(summary_csv_path, output_path=None, output_name=None, compare_summary_csv_path=None, real_values=False):
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
    
    summary_paths = [Path(summary_csv_path)]
    if compare_summary_csv_path is not None:
        summary_paths.append(Path(compare_summary_csv_path))

    for summary_path in summary_paths:
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    df_summaries = [pd.read_csv(summary_path) for summary_path in summary_paths]
    for summary_path, df_summary in zip(summary_paths, df_summaries):
        print(f"✓ Loaded summary from {summary_path}")
        print(f"  Shape: {df_summary.shape}")
        print(f"  Summary types: {df_summary['summary_type'].unique().tolist()}")
    
    # Extract ring IDs from columns
    ring_ids = _ring_ids_from_columns(df_summaries[0].columns, 'physical_ring')
    print(f"  Number of rings: {len(ring_ids)}")

    summary_types = _ordered_summary_types(
        sorted({summary_type for df_summary in df_summaries for summary_type in df_summary['summary_type'].dropna().unique().tolist()})
    )
    for summary_path, df_summary in zip(summary_paths, df_summaries):
        print(f"  Aggregating over {len(df_summary)} rows and {df_summary['complex_name'].nunique() if 'complex_name' in df_summary.columns else 'unknown'} complexes from {summary_path.name}")

    source_labels = [_display_name_from_summary_path(summary_path) for summary_path in summary_paths]
    print(f"  Plot sources: {source_labels}")
    plot_title = ', '.join(source_labels)

    if real_values:
        for summary_path, df_summary in zip(summary_paths, df_summaries):
            if 'radius' not in df_summary.columns:
                raise ValueError(f'radius column is required for real-values plotting: {summary_path}')

        real_physical_data = [
            _aggregate_real_metric_by_summary_type(df_summary, summary_types, ring_ids, 'physical')
            for df_summary in df_summaries
        ]
        real_zernike_data = [
            _aggregate_real_metric_by_summary_type(df_summary, summary_types, ring_ids, 'zernike')
            for df_summary in df_summaries
        ]
        for summary_type in summary_types:
            for source_index, source_label in enumerate(source_labels):
                counts = [real_physical_data[source_index][summary_type]['bin_values'][bid]['count'] for bid in range(25)]
                print(f"  real / {source_label} / {summary_type}: bin sample counts {counts}")

        bin_centers = real_physical_data[0][summary_types[0]]['bin_centers']
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

        _plot_real_metric_panel(
            axes[0],
            real_physical_data,
            bin_centers,
            summary_types,
            'Real Values - Physical Distance per Bin',
            'Physical Distance (Å)',
            source_labels=source_labels,
        )
        _plot_real_metric_panel(
            axes[1],
            real_zernike_data,
            bin_centers,
            summary_types,
            'Real Values - Zernike Distance per Bin',
            'Zernike Distance',
            source_labels=source_labels,
        )

        fig.suptitle(f'Real Radial Values with Uncertainties ({plot_title})', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()

        real_output_path = _resolve_real_output_path(summary_paths[0], output_path, output_name)
        fig.savefig(real_output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Real-values plot saved to {real_output_path}")

        plt.close(fig)
        return

    plot_modes = [
        ('raw', None, 'Raw ring values'),
        ('max_norm', 'max', 'Per-complex max-normalized'),
        ('minmax_norm', 'minmax', 'Per-complex min-max normalized'),
    ]

    panel_data = {}
    for mode_name, normalization_mode, mode_label in plot_modes:
        physical_data = [
            _aggregate_metric_by_summary_type(df_summary, summary_types, ring_ids, 'physical', normalization_mode=normalization_mode)
            for df_summary in df_summaries
        ]
        zernike_data = [
            _aggregate_metric_by_summary_type(df_summary, summary_types, ring_ids, 'zernike', normalization_mode=normalization_mode)
            for df_summary in df_summaries
        ]
        panel_data[mode_name] = {
            'label': mode_label,
            'physical': physical_data,
            'zernike': zernike_data,
        }
        for summary_index, summary_type in enumerate(summary_types):
            for source_index, source_label in enumerate(source_labels):
                counts = [panel_data[mode_name]['physical'][source_index][summary_type][rid]['count'] for rid in ring_ids]
                print(f"  {mode_name} / {source_label} / {summary_type}: ring sample counts {counts}")

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
        physical_ylabel = 'Physical Distance (Å)' if mode_name == 'raw' else 'Physical Distance'
        _plot_metric_panel(
            axes[row_index, 0],
            panel_data[mode_name]['physical'],
            ring_ids,
            summary_types,
            colors,
            markers,
            f'{mode_label} - Physical Distance per Ring',
            physical_ylabel,
            source_labels=source_labels,
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
            source_labels=source_labels,
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

    fig.suptitle(f'Radial Functions with Uncertainties ({plot_title})',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    # Save output
    output_path = _resolve_output_path(summary_paths[0], output_path, output_name)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot radial function histograms with uncertainties from summary CSV'
    )
    parser.add_argument('summary_csv', help='Path to summary.csv from get_complementary_plane2.py')
    parser.add_argument('--compare-summary-csv', help='Optional second summary CSV to plot on the same figure')
    parser.add_argument('-o', '--output', help='Output path for plot (default: <stem>_radial_histograms.png)')
    parser.add_argument('--output-name', help='Output file name without extension (default: <summary>_radial_histograms)')
    parser.add_argument('--real-values', action='store_true', help='Also create a plot that bins rings by their real radial distance')
    
    args = parser.parse_args()
    
    try:
        plot_radial_histograms(args.summary_csv, args.output, args.output_name, args.compare_summary_csv, real_values=args.real_values)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
