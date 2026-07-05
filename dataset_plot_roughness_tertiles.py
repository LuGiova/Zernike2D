#!/usr/bin/env python3
"""Plot roughness quantiles split by feature tertiles.

This is the roughness-only counterpart of dataset_plot_tertiles.py. It
builds one figure per feature with a distribution histogram on top and two
rows of ring profiles below it: classical ring IDs on the left and real
radial distances on the right.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


FEATURES = ['gyration_radius', 'radius']
TERTILES = ['low', 'medium', 'large']
SUMMARY_ORDER = ['weighted', 'normal']
N_RINGS = 10

TERTILE_COLORS = {
    'low': '#1b9e77',
    'medium': '#d95f02',
    'large': '#7570b3',
}
FILE_COLORS = ['#1f77b4', '#ff7f0e']
SUMMARY_STYLES = {
    'weighted': {'marker': 'o', 'linestyle': '-'},
    'normal': {'marker': 's', 'linestyle': '--'},
}


def _ordered_summary_types(values):
    summary_types = [summary_type for summary_type in SUMMARY_ORDER if summary_type in values]
    for summary_type in values:
        if summary_type not in summary_types:
            summary_types.append(summary_type)
    return summary_types


def _ring_ids_from_columns(columns, prefix):
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)(?:_(?:mean|uncertainty))?$')
    ring_ids = set()
    for column in columns:
        match = pattern.match(column)
        if match is not None:
            ring_ids.add(int(match.group(1)))
    if not ring_ids:
        raise ValueError(f'No columns found for prefix {prefix!r}')
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


def _display_name_from_summary_path(summary_path):
    name = Path(summary_path).name
    if name.endswith('_summary.csv'):
        return name[:-len('_summary.csv')]
    stem = Path(name).stem
    return stem[:-len('_summary')] if stem.endswith('_summary') else stem


def _parse_quantiles(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError('quantiles must be a Python list or tuple of three probabilities') from exc

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise argparse.ArgumentTypeError('quantiles must contain exactly three probabilities')

    probabilities = [float(entry) for entry in parsed]
    if any(probability < 0 for probability in probabilities):
        raise argparse.ArgumentTypeError('quantiles must be non-negative')

    total = float(sum(probabilities))
    if not np.isclose(total, 1.0):
        raise argparse.ArgumentTypeError('quantiles must sum to 1')

    return probabilities


def _resolve_feature_output_path(summary_path, feature_name, output_path=None, output_name=None, compare_summary_path=None):
    summary_path = Path(summary_path)
    if output_name:
        base_stem = output_name
    elif compare_summary_path is not None:
        base_stem = f'{_display_name_from_summary_path(summary_path)}_vs_{_display_name_from_summary_path(compare_summary_path)}'
    else:
        base_stem = _display_name_from_summary_path(summary_path)

    filename = f'{base_stem}_{feature_name}_tertiles.pdf'

    if output_path is None:
        return summary_path.parent / filename

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / filename
    if output_path.suffix:
        return output_path.with_name(f'{filename[:-4]}{output_path.suffix}')
    return output_path / filename


def _finite_feature_values(df_summary, feature_name):
    if 'complex_name' not in df_summary.columns:
        raise ValueError('complex_name column is required to compute tertiles')
    if feature_name not in df_summary.columns:
        raise ValueError(f'{feature_name} column is required')

    feature_values = []
    for complex_name, complex_group in df_summary.groupby('complex_name', sort=False):
        values = pd.to_numeric(complex_group[feature_name], errors='coerce').to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        feature_values.append((complex_name, float(values[0])))

    if not feature_values:
        return pd.DataFrame(columns=['complex_name', feature_name])

    return pd.DataFrame(feature_values, columns=['complex_name', feature_name])


def _quantile_thresholds(feature_values, probabilities=None):
    if len(feature_values) == 0:
        return float('nan'), float('nan')
    if probabilities is None:
        probabilities = [1 / 3, 1 / 3, 1 / 3]
    cumulative_probabilities = np.cumsum(np.asarray(probabilities, dtype=float))[:-1]
    q1, q2 = np.quantile(feature_values, cumulative_probabilities)
    return float(q1), float(q2)


def _tertile_from_value(value, low_threshold, high_threshold):
    if not np.isfinite(value):
        return None
    if value <= low_threshold:
        return 'low'
    if value <= high_threshold:
        return 'medium'
    return 'large'


def _complex_names_by_tertile(df_summary, feature_name, low_threshold, high_threshold):
    feature_df = _finite_feature_values(df_summary, feature_name)
    tertile_map = {tertile: [] for tertile in TERTILES}
    if feature_df.empty:
        return tertile_map

    for _, row in feature_df.iterrows():
        tertile = _tertile_from_value(row[feature_name], low_threshold, high_threshold)
        if tertile is not None:
            tertile_map[tertile].append(row['complex_name'])
    return tertile_map


def _aggregate_roughness_by_summary_type(df_summary, summary_types, ring_ids, complex_names=None):
    aggregated = {}
    complex_name_filter = set(complex_names) if complex_names is not None else None

    for summary_type in summary_types:
        group = df_summary[df_summary['summary_type'] == summary_type]
        per_ring_values = {rid: [] for rid in ring_ids}

        if 'complex_name' not in group.columns:
            raise ValueError('complex_name column is required to aggregate by complex')

        for complex_name, complex_group in group.groupby('complex_name', sort=False):
            if complex_name_filter is not None and complex_name not in complex_name_filter:
                continue

            for rid in ring_ids:
                mean_col = f'roughness_ring{rid}'
                if mean_col not in complex_group.columns:
                    continue

                values = pd.to_numeric(complex_group[mean_col], errors='coerce').to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    per_ring_values[rid].append(float(values[0]))

        ring_values = {}
        for rid in ring_ids:
            finite_means = np.array(per_ring_values[rid], dtype=float)
            if len(finite_means) == 0:
                ring_values[rid] = {
                    'mean': float('nan'),
                    'uncertainty': float('nan'),
                    'count': 0,
                }
                continue

            mean_value, sample_uncertainty = _mean_and_sample_uncertainty(finite_means)
            ring_values[rid] = {
                'mean': mean_value,
                'uncertainty': sample_uncertainty,
                'count': int(len(finite_means)),
            }

        aggregated[summary_type] = ring_values

    return aggregated


def _aggregate_real_roughness_by_summary_type(
    df_summary,
    summary_types,
    ring_ids,
    complex_names=None,
    bin_width=2.0,
    max_x=50.0,
):
    aggregated = {}
    complex_name_filter = set(complex_names) if complex_names is not None else None
    divisor = float(len(ring_ids) if ring_ids else N_RINGS)
    bin_edges = np.arange(0.0, max_x + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2.0
    bin_ids = list(range(len(bin_centers)))

    for summary_type in summary_types:
        group = df_summary[df_summary['summary_type'] == summary_type]
        per_bin_values = {bid: [] for bid in bin_ids}

        if 'complex_name' not in group.columns:
            raise ValueError('complex_name column is required to aggregate by complex')
        if 'radius' not in group.columns:
            raise ValueError('radius column is required to compute real-valued ring positions')

        for complex_name, complex_group in group.groupby('complex_name', sort=False):
            if complex_name_filter is not None and complex_name not in complex_name_filter:
                continue

            radius_values = pd.to_numeric(complex_group['radius'], errors='coerce').to_numpy(dtype=float)
            radius_values = radius_values[np.isfinite(radius_values)]
            if len(radius_values) == 0:
                continue

            radius = float(radius_values[0])
            if not np.isfinite(radius) or radius <= 0:
                continue

            for rid in ring_ids:
                mean_col = f'roughness_ring{rid}'
                if mean_col not in complex_group.columns:
                    continue

                values = pd.to_numeric(complex_group[mean_col], errors='coerce').to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue

                x_position = (rid - 0.5) * radius / divisor
                if x_position > max_x:
                    continue

                bin_index = int(np.floor(x_position / bin_width))
                bin_index = min(bin_index, len(bin_ids) - 1)
                per_bin_values[bin_index].append(float(values[0]))

        bin_values = {}
        for bid in bin_ids:
            finite_means = np.array(per_bin_values[bid], dtype=float)
            if len(finite_means) == 0:
                bin_values[bid] = {
                    'mean': float('nan'),
                    'uncertainty': float('nan'),
                    'count': 0,
                }
                continue

            mean_value, sample_uncertainty = _mean_and_sample_uncertainty(finite_means)
            bin_values[bid] = {
                'mean': mean_value,
                'uncertainty': sample_uncertainty,
                'count': int(len(finite_means)),
            }

        aggregated[summary_type] = {
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'bin_values': bin_values,
        }

    return aggregated


def _aggregate_feature_histogram_values(df_summary, feature_name):
    feature_df = _finite_feature_values(df_summary, feature_name)
    return feature_df[feature_name].to_numpy(dtype=float)


def _feature_display_label(feature_name):
    if feature_name == 'gyration_radius':
        return 'Gyration Radius (Å)'
    if feature_name == 'radius':
        return 'Radius (Å)'
    if feature_name == 'roughness':
        return 'Roughness (Å)'
    return feature_name


def _hist_bins(*series):
    values = [np.asarray(series_values, dtype=float) for series_values in series if len(series_values) > 0]
    if not values:
        return 10

    combined = np.concatenate(values)
    if len(combined) < 2:
        return 10

    min_value = float(np.min(combined))
    max_value = float(np.max(combined))
    if np.isclose(min_value, max_value):
        return 10

    return np.histogram_bin_edges(combined, bins='auto')


def _quantile_legend_label(probabilities=None):
    if probabilities is None:
        return 'quantile threshold'

    percentages = ', '.join(f'{probability * 100:.1f}%' for probability in probabilities)
    return f'quantile thresholds ({percentages})'


def _source_tertile_legend_label(source_label, feature_values, low_threshold, high_threshold):
    finite_values = np.asarray(feature_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return f'{source_label}: no data'

    low_count = int(np.sum(finite_values <= low_threshold))
    medium_count = int(np.sum((finite_values > low_threshold) & (finite_values <= high_threshold)))
    large_count = int(np.sum(finite_values > high_threshold))
    total = low_count + medium_count + large_count
    if total == 0:
        return f'{source_label}: no data'

    low_pct = low_count / total * 100.0
    medium_pct = medium_count / total * 100.0
    large_pct = large_count / total * 100.0
    return f'{source_label}: low {low_pct:.1f}%, medium {medium_pct:.1f}%, large {large_pct:.1f}%'


def _plot_hist_panel(ax, hist_values_by_source, source_labels, thresholds, feature_name, quantiles=None):
    colors = FILE_COLORS[: len(hist_values_by_source)]
    bins = _hist_bins(*hist_values_by_source)
    threshold_label = _quantile_legend_label(quantiles)
    display_label = _feature_display_label(feature_name)

    for source_values, source_label, color in zip(hist_values_by_source, source_labels, colors):
        ax.hist(
            source_values,
            bins=bins,
            color=color,
            alpha=0.5 if len(hist_values_by_source) > 1 else 0.75,
            edgecolor='black',
            linewidth=0.8,
            label=source_label,
        )

    for threshold_index, threshold in enumerate(thresholds):
        if np.isfinite(threshold):
            ax.axvline(
                threshold,
                color='black',
                linestyle='--',
                linewidth=1.2,
                label=threshold_label if threshold_index == 0 else None,
            )

    ax.set_xlabel(display_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{display_label} distribution and quantile thresholds', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.legend(fontsize=10, loc='best')


def _plot_series(ax, x_positions, means, uncertainties, color, label, marker, linestyle):
    ax.errorbar(
        x_positions,
        means,
        yerr=uncertainties,
        fmt='none',
        ecolor=color,
        elinewidth=1.5,
        capsize=3,
        capthick=1.2,
        zorder=2,
    )
    ax.plot(
        x_positions,
        means,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.4,
        markersize=4.5,
        label=label,
        zorder=3,
    )


def _plot_roughness_panel(
    ax,
    panel_data,
    x_positions,
    x_tick_labels,
    summary_types,
    tertile,
    source_labels,
    x_label,
    panel_title,
    source_mode='tertile',
):
    datasets = []
    for source_index, source_label in enumerate(source_labels):
        ring_values = panel_data[source_index]
        for summary_type in summary_types:
            values = ring_values[summary_type]
            if 'bin_values' in values:
                values = values['bin_values']
            keys = list(values.keys())
            means = np.array([values[key]['mean'] for key in keys], dtype=float)
            uncertainties = np.array([values[key]['uncertainty'] for key in keys], dtype=float)
            datasets.append((source_index, source_label, summary_type, means, uncertainties))

    offset_step = 0.18 / max(len(datasets), 1)
    for index, (source_index, source_label, summary_type, means, uncertainties) in enumerate(datasets):
        x_offset = (index - (len(datasets) - 1) / 2) * offset_step
        style = SUMMARY_STYLES.get(summary_type, {'marker': 'o', 'linestyle': '-'})
        if source_mode == 'file':
            color = FILE_COLORS[source_index % len(FILE_COLORS)]
        else:
            color = TERTILE_COLORS[tertile]

        _plot_series(
            ax,
            x_positions + x_offset,
            means,
            uncertainties,
            color,
            f'{source_label} / {summary_type}' if len(source_labels) > 1 else summary_type,
            style['marker'],
            style['linestyle'],
        )

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Roughness (Å)', fontsize=12, fontweight='bold')
    ax.set_title(panel_title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if len(source_labels) > 1:
        file_handles = [
            Patch(facecolor=FILE_COLORS[source_index % len(FILE_COLORS)], edgecolor='black', label=source_label)
            for source_index, source_label in enumerate(source_labels)
        ]
        legend1 = ax.legend(handles=file_handles, title='file', fontsize=9, loc='upper left')
        ax.add_artist(legend1)

    style_handles = []
    for summary_type in summary_types:
        style = SUMMARY_STYLES.get(summary_type, {'marker': 'o', 'linestyle': '-'})
        style_handles.append(
            Line2D(
                [0],
                [0],
                color='#444444',
                marker=style['marker'],
                linestyle=style['linestyle'],
                label=summary_type,
            )
        )
    ax.legend(handles=style_handles, title='summary', fontsize=9, loc='lower left')


def _build_tertile_roughness_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold):
    tertile_complex_names = _complex_names_by_tertile(df_summary, feature_name, low_threshold, high_threshold)
    tertile_data = {}
    for tertile in TERTILES:
        tertile_data[tertile] = _aggregate_roughness_by_summary_type(
            df_summary,
            summary_types,
            ring_ids,
            complex_names=tertile_complex_names[tertile],
        )
    return tertile_data


def _build_tertile_real_roughness_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold):
    tertile_complex_names = _complex_names_by_tertile(df_summary, feature_name, low_threshold, high_threshold)
    tertile_data = {}
    for tertile in TERTILES:
        tertile_data[tertile] = _aggregate_real_roughness_by_summary_type(
            df_summary,
            summary_types,
            ring_ids,
            complex_names=tertile_complex_names[tertile],
        )
    return tertile_data


def plot_feature_tertile_histograms(summary_csv_path, output_path=None, output_name=None, compare_summary_csv_path=None, quantiles=None):
    summary_paths = [Path(summary_csv_path)]
    if compare_summary_csv_path is not None:
        summary_paths.append(Path(compare_summary_csv_path))

    for summary_path in summary_paths:
        if not summary_path.exists():
            raise FileNotFoundError(f'Summary file not found: {summary_path}')

    df_summaries = [pd.read_csv(summary_path) for summary_path in summary_paths]
    for summary_path, df_summary in zip(summary_paths, df_summaries):
        print(f'✓ Loaded summary from {summary_path}')
        print(f'  Shape: {df_summary.shape}')
        print(f"  Summary types: {df_summary['summary_type'].dropna().unique().tolist()}")

    ring_ids = _ring_ids_from_columns(df_summaries[0].columns, 'roughness_ring')
    summary_types = _ordered_summary_types(
        sorted({summary_type for df_summary in df_summaries for summary_type in df_summary['summary_type'].dropna().unique().tolist()})
    )
    source_labels = [_display_name_from_summary_path(summary_path) for summary_path in summary_paths]
    print(f'  Number of rings: {len(ring_ids)}')
    print(f'  Plot sources: {source_labels}')

    for feature_name in FEATURES:
        display_label = _feature_display_label(feature_name)
        feature_values_by_source = [_aggregate_feature_histogram_values(df_summary, feature_name) for df_summary in df_summaries]
        combined_feature_values = (
            np.concatenate([values for values in feature_values_by_source if len(values) > 0])
            if any(len(values) > 0 for values in feature_values_by_source)
            else np.array([], dtype=float)
        )
        low_threshold, high_threshold = _quantile_thresholds(combined_feature_values, quantiles)
        print(f'  {feature_name}: q1={low_threshold:.6g}, q2={high_threshold:.6g}')
        histogram_source_labels = [
            _source_tertile_legend_label(source_label, source_values, low_threshold, high_threshold)
            for source_label, source_values in zip(source_labels, feature_values_by_source)
        ]

        tertile_data_by_source = [
            _build_tertile_roughness_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold)
            for df_summary in df_summaries
        ]
        real_tertile_data_by_source = [
            _build_tertile_real_roughness_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold)
            for df_summary in df_summaries
        ]

        fig = plt.figure(figsize=(18, 18))
        gs = fig.add_gridspec(4, 2, height_ratios=[0.9, 1.2, 1.2, 1.2])
        ax_hist = fig.add_subplot(gs[0, :])
        _plot_hist_panel(
            ax_hist,
            feature_values_by_source,
            histogram_source_labels,
            (low_threshold, high_threshold),
            feature_name,
            quantiles,
        )

        if len(ring_ids) == 0:
            raise ValueError('No roughness ring columns found')

        x_positions_classic = np.arange(len(ring_ids))
        x_tick_labels_classic = [f'{ring_id}' for ring_id in ring_ids]
        x_positions_real = np.arange(25) * 2.0 + 1.0
        x_tick_labels_real = [f'{center:.0f}' for center in x_positions_real]

        for row_index, tertile in enumerate(TERTILES, start=1):
            ax_classic = fig.add_subplot(gs[row_index, 0])
            ax_real = fig.add_subplot(gs[row_index, 1])

            _plot_roughness_panel(
                ax_classic,
                [tertile_data_by_source[file_index][tertile] for file_index in range(len(df_summaries))],
                x_positions_classic,
                x_tick_labels_classic,
                summary_types,
                tertile,
                source_labels,
                'Ring ID',
                f'{display_label} - {tertile.capitalize()} quantile - classical ring positions',
                source_mode='file' if len(df_summaries) > 1 else 'tertile',
            )
            _plot_roughness_panel(
                ax_real,
                [real_tertile_data_by_source[file_index][tertile] for file_index in range(len(df_summaries))],
                x_positions_real,
                x_tick_labels_real,
                summary_types,
                tertile,
                source_labels,
                'Real radial distance bin center (Å)',
                f'{display_label} - {tertile.capitalize()} quantile - real radial positions',
                source_mode='file' if len(df_summaries) > 1 else 'tertile',
            )

            ax_classic.text(
                -0.18,
                0.5,
                tertile,
                transform=ax_classic.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=12,
                fontweight='bold',
            )

        fig.suptitle(
            f'{display_label} quantiles with roughness profiles ({", ".join(source_labels)})',
            fontsize=15,
            fontweight='bold',
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))

        output_file = _resolve_feature_output_path(
            summary_paths[0],
            feature_name,
            output_path,
            output_name,
            compare_summary_path=summary_paths[1] if len(summary_paths) > 1 else None,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'✓ Saved {feature_name} plot to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot roughness grouped by feature tertiles with classical and real-value ring profiles.'
    )
    parser.add_argument('summary_csv', help='Path to a roughness summary CSV file')
    parser.add_argument('--compare-summary-csv', help='Optional second roughness summary CSV to plot alongside the first one')
    parser.add_argument('-o', '--output', help='Output path or directory for the plots')
    parser.add_argument('--output-name', help='Output file name prefix without extension')
    parser.add_argument('--quantiles', type=_parse_quantiles, help='Three probabilities that sum to 1, e.g. [0.4, 0.4, 0.2]')

    args = parser.parse_args()

    try:
        plot_feature_tertile_histograms(
            args.summary_csv,
            args.output,
            args.output_name,
            args.compare_summary_csv,
            args.quantiles,
        )
    except Exception as e:
        print(f'✗ Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()