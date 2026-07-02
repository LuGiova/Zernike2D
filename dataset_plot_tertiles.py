#!/usr/bin/env python3
"""Plot RDFs split by feature quantiles for summary CSV files.

The script keeps the same input style as dataset_plot.py, but instead of
normalization modes it splits complexes into low/medium/high quantiles for
gyration_radius, flatness, and radius.
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


FEATURES = ['gyration_radius', 'flatness', 'radius']
FEATURES = ['gyration_radius', 'flatness', 'radius', 'roughness']
TERTILES = ['low', 'medium', 'large']
SUMMARY_ORDER = ['weighted', 'normal']

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
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)_(?:mean|uncertainty)$')
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


def _resolve_feature_output_path(
    summary_path,
    feature_name,
    output_path=None,
    output_name=None,
    compare_summary_path=None,
    real_values=False,
):
    summary_path = Path(summary_path)
    if output_name:
        base_stem = output_name
    elif compare_summary_path is not None:
        base_stem = f'{_display_name_from_summary_path(summary_path)}_vs_{_display_name_from_summary_path(compare_summary_path)}'
    else:
        base_stem = _display_name_from_summary_path(summary_path)

    suffix = 'real_rdf' if real_values else 'mean_rdf'
    filename = f'{base_stem}_{feature_name}_{suffix}.pdf'

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
    q1 = float(q1)
    q2 = float(q2)
    return q1, q2


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


def _aggregate_metric_by_summary_type(df_summary, summary_types, ring_ids, metric_prefix, complex_names=None):
    aggregated = {}
    complex_name_filter = set(complex_names) if complex_names is not None else None

    for summary_type in summary_types:
        group = df_summary[df_summary['summary_type'] == summary_type]
        ring_values = {}
        per_ring_values = {rid: [] for rid in ring_ids}
        per_ring_uncertainties = {rid: [] for rid in ring_ids}

        if 'complex_name' not in group.columns:
            raise ValueError('complex_name column is required to aggregate by complex')

        for complex_name, complex_group in group.groupby('complex_name', sort=False):
            if complex_name_filter is not None and complex_name not in complex_name_filter:
                continue

            complex_means = []
            complex_uncertainties = []

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

            for rid, value, uncertainty in zip(ring_ids, complex_means, complex_uncertainties):
                if np.isfinite(value):
                    per_ring_values[rid].append(value)
                if np.isfinite(uncertainty):
                    per_ring_uncertainties[rid].append(uncertainty)

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
    complex_names=None,
    bin_width=2.0,
    max_x=50.0,
):
    aggregated = {}
    complex_name_filter = set(complex_names) if complex_names is not None else None
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

            for idx, x_position in enumerate(complex_x_positions):
                if not np.isfinite(x_position) or x_position > max_x:
                    continue

                bin_index = int(np.floor(x_position / bin_width))
                bin_index = min(bin_index, len(bin_ids) - 1)

                if np.isfinite(complex_means[idx]):
                    per_bin_values[bin_index].append(complex_means[idx])
                if np.isfinite(complex_uncertainties[idx]):
                    per_bin_uncertainties[bin_index].append(complex_uncertainties[idx])

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


def _aggregate_feature_histogram_values(df_summary, feature_name):
    feature_df = _finite_feature_values(df_summary, feature_name)
    return feature_df[feature_name].to_numpy(dtype=float)


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


def _plot_hist_panel(ax, hist_values_by_source, source_labels, thresholds, feature_name):
    colors = FILE_COLORS[: len(hist_values_by_source)]
    bins = _hist_bins(*hist_values_by_source)

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
                label='quantile threshold' if threshold_index == 0 else None,
            )

    ax.set_xlabel(feature_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{feature_name} distribution and quantile thresholds', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.legend(fontsize=10, loc='best')


def _plot_series(ax, x_positions, means, uncertainties, color, label, marker, linestyle, uncertainty_scale=3.0):
    uncertainties = np.asarray(uncertainties, dtype=float) * uncertainty_scale
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


def _plot_single_file_rdf_panel(
    ax,
    tertile_data,
    x_positions,
    x_tick_labels,
    summary_types,
    metric_label,
    value_label,
    x_label,
    uncertainty_scale=3.0,
):
    series = []
    for tertile in TERTILES:
        for summary_type in summary_types:
            ring_values = tertile_data[tertile][summary_type]
            if 'bin_values' in ring_values:
                ring_values = ring_values['bin_values']
            keys = list(ring_values.keys())
            means = np.array([ring_values[key]['mean'] for key in keys], dtype=float)
            uncertainties = np.array([ring_values[key]['combined_uncertainty'] for key in keys], dtype=float)
            series.append((tertile, summary_type, means, uncertainties))

    offset_step = 0.16 / max(len(series), 1)
    for index, (tertile, summary_type, means, uncertainties) in enumerate(series):
        x_offset = (index - (len(series) - 1) / 2) * offset_step
        style = SUMMARY_STYLES.get(summary_type, {'marker': 'o', 'linestyle': '-'})
        _plot_series(
            ax,
            x_positions + x_offset,
            means,
            uncertainties,
            TERTILE_COLORS[tertile],
            f'{tertile} / {summary_type}',
            style['marker'],
            style['linestyle'],
            uncertainty_scale=uncertainty_scale,
        )

    ax.set_xlabel('Ring ID', fontsize=12, fontweight='bold')
    ax.set_ylabel(value_label, fontsize=12, fontweight='bold')
    ax.set_title(metric_label, fontsize=13, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    tertile_handles = [
        Patch(facecolor=TERTILE_COLORS[tertile], edgecolor='black', label=tertile)
        for tertile in TERTILES
    ]
    legend1 = ax.legend(handles=tertile_handles, title='quantile', fontsize=9, loc='upper left')
    ax.add_artist(legend1)


def _plot_dual_file_rdf_panel(
    ax,
    file_data,
    x_positions,
    x_tick_labels,
    summary_types,
    metric_label,
    value_label,
    file_labels,
    x_label,
    uncertainty_scale=3.0,
):
    series = []
    for file_index, file_label in enumerate(file_labels):
        for summary_type in summary_types:
            ring_values = file_data[file_index][summary_type]
            if 'bin_values' in ring_values:
                ring_values = ring_values['bin_values']
            keys = list(ring_values.keys())
            means = np.array([ring_values[key]['mean'] for key in keys], dtype=float)
            uncertainties = np.array([ring_values[key]['combined_uncertainty'] for key in keys], dtype=float)
            series.append((file_index, file_label, summary_type, means, uncertainties))

    offset_step = 0.18 / max(len(series), 1)
    for index, (file_index, file_label, summary_type, means, uncertainties) in enumerate(series):
        x_offset = (index - (len(series) - 1) / 2) * offset_step
        style = SUMMARY_STYLES.get(summary_type, {'marker': 'o', 'linestyle': '-'})
        _plot_series(
            ax,
            x_positions + x_offset,
            means,
            uncertainties,
            FILE_COLORS[file_index % len(FILE_COLORS)],
            f'{file_label} / {summary_type}',
            style['marker'],
            style['linestyle'],
            uncertainty_scale=uncertainty_scale,
        )

    ax.set_xlabel('Ring ID', fontsize=12, fontweight='bold')
    ax.set_ylabel(value_label, fontsize=12, fontweight='bold')
    ax.set_title(metric_label, fontsize=13, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    file_handles = [
        Patch(facecolor=FILE_COLORS[file_index % len(FILE_COLORS)], edgecolor='black', label=file_label)
        for file_index, file_label in enumerate(file_labels)
    ]
    legend1 = ax.legend(handles=file_handles, title='file', fontsize=9, loc='upper left')
    ax.add_artist(legend1)


def _build_tertile_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold):
    tertile_complex_names = _complex_names_by_tertile(df_summary, feature_name, low_threshold, high_threshold)
    tertile_data = {}
    for tertile in TERTILES:
        tertile_data[tertile] = {
            'physical': _aggregate_metric_by_summary_type(
                df_summary,
                summary_types,
                ring_ids,
                'physical',
                complex_names=tertile_complex_names[tertile],
            ),
            'zernike': _aggregate_metric_by_summary_type(
                df_summary,
                summary_types,
                ring_ids,
                'zernike',
                complex_names=tertile_complex_names[tertile],
            ),
        }
    return tertile_data


def _build_tertile_real_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold):
    tertile_complex_names = _complex_names_by_tertile(df_summary, feature_name, low_threshold, high_threshold)
    tertile_data = {}
    for tertile in TERTILES:
        tertile_data[tertile] = {
            'physical': _aggregate_real_metric_by_summary_type(
                df_summary,
                summary_types,
                ring_ids,
                'physical',
                complex_names=tertile_complex_names[tertile],
            ),
            'zernike': _aggregate_real_metric_by_summary_type(
                df_summary,
                summary_types,
                ring_ids,
                'zernike',
                complex_names=tertile_complex_names[tertile],
            ),
        }
    return tertile_data


def plot_feature_tertile_histograms(
    summary_csv_path,
    output_path=None,
    output_name=None,
    compare_summary_csv_path=None,
    real_values=False,
    quantiles=None,
):
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

    ring_ids = _ring_ids_from_columns(df_summaries[0].columns, 'physical_ring')
    summary_types = _ordered_summary_types(
        sorted({summary_type for df_summary in df_summaries for summary_type in df_summary['summary_type'].dropna().unique().tolist()})
    )
    source_labels = [_display_name_from_summary_path(summary_path) for summary_path in summary_paths]
    print(f'  Number of rings: {len(ring_ids)}')
    print(f'  Plot sources: {source_labels}')

    for feature_name in FEATURES:
        feature_values_by_source = [_aggregate_feature_histogram_values(df_summary, feature_name) for df_summary in df_summaries]
        combined_feature_values = np.concatenate([values for values in feature_values_by_source if len(values) > 0]) if any(len(values) > 0 for values in feature_values_by_source) else np.array([], dtype=float)
        low_threshold, high_threshold = _quantile_thresholds(combined_feature_values, quantiles)
        print(f'  {feature_name}: q1={low_threshold:.6g}, q2={high_threshold:.6g}')

        if real_values:
            tertile_data_by_source = [
                _build_tertile_real_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold)
                for df_summary in df_summaries
            ]
            x_positions = np.arange(25) * 2.0 + 1.0
            x_tick_labels = [f'{center:.0f}' for center in x_positions]
            x_label = 'Real radial distance bin center (Å)'
            uncertainty_scale = 1.0
        else:
            tertile_data_by_source = [
                _build_tertile_data(df_summary, summary_types, ring_ids, feature_name, low_threshold, high_threshold)
                for df_summary in df_summaries
            ]
            x_positions = np.arange(len(ring_ids))
            x_tick_labels = [f'{r}' for r in ring_ids]
            x_label = 'Ring ID'
            uncertainty_scale = 3.0

        if len(df_summaries) == 1:
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.4])
            ax_hist = fig.add_subplot(gs[0, :])
            ax_physical = fig.add_subplot(gs[1, 0])
            ax_zernike = fig.add_subplot(gs[1, 1])

            _plot_hist_panel(ax_hist, feature_values_by_source, source_labels, (low_threshold, high_threshold), feature_name)

            _plot_single_file_rdf_panel(
                ax_physical,
                {tertile: tertile_data_by_source[0][tertile]['physical'] for tertile in TERTILES},
                x_positions,
                x_tick_labels,
                summary_types,
                f'{feature_name} - Physical Distance per Ring',
                'Physical Distance (Å)',
                x_label,
                uncertainty_scale=uncertainty_scale,
            )
            _plot_single_file_rdf_panel(
                ax_zernike,
                {tertile: tertile_data_by_source[0][tertile]['zernike'] for tertile in TERTILES},
                x_positions,
                x_tick_labels,
                summary_types,
                f'{feature_name} - Zernike Distance per Ring',
                'Zernike Distance',
                x_label,
                uncertainty_scale=uncertainty_scale,
            )

            fig.suptitle(f'{feature_name} quantiles with RDFs ({source_labels[0]})', fontsize=15, fontweight='bold', y=0.995)
        else:
            fig = plt.figure(figsize=(18, 18))
            gs = fig.add_gridspec(4, 2, height_ratios=[0.9, 1.2, 1.2, 1.2])
            ax_hist = fig.add_subplot(gs[0, :])
            _plot_hist_panel(ax_hist, feature_values_by_source, source_labels, (low_threshold, high_threshold), feature_name)

            for row_index, tertile in enumerate(TERTILES, start=1):
                ax_physical = fig.add_subplot(gs[row_index, 0])
                ax_zernike = fig.add_subplot(gs[row_index, 1])

                _plot_dual_file_rdf_panel(
                    ax_physical,
                    [tertile_data_by_source[file_index][tertile]['physical'] for file_index in range(len(df_summaries))],
                    x_positions,
                    x_tick_labels,
                    summary_types,
                    f'{feature_name} - {tertile.capitalize()} quantile - Physical Distance per Ring',
                    'Physical Distance (Å)',
                    source_labels,
                    x_label,
                    uncertainty_scale=uncertainty_scale,
                )
                _plot_dual_file_rdf_panel(
                    ax_zernike,
                    [tertile_data_by_source[file_index][tertile]['zernike'] for file_index in range(len(df_summaries))],
                    x_positions,
                    x_tick_labels,
                    summary_types,
                    f'{feature_name} - {tertile.capitalize()} quantile - Zernike Distance per Ring',
                    'Zernike Distance',
                    source_labels,
                    x_label,
                    uncertainty_scale=uncertainty_scale,
                )

                ax_physical.text(
                    -0.18,
                    0.5,
                    tertile,
                    transform=ax_physical.transAxes,
                    rotation=90,
                    va='center',
                    ha='center',
                    fontsize=12,
                    fontweight='bold',
                )

            fig.suptitle(
                f'{feature_name} quantiles with RDFs ({", ".join(source_labels)})',
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
            real_values=real_values,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'✓ Saved {feature_name} plot to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot RDFs grouped by quantiles of gyration_radius, flatness, and radius.'
    )
    parser.add_argument('summary_csv', help='Path to summary.csv from get_complementary_plane2.py')
    parser.add_argument('--compare-summary-csv', help='Optional second summary CSV to plot alongside the first one')
    parser.add_argument('-o', '--output', help='Output path or directory for the plots')
    parser.add_argument('--output-name', help='Output file name prefix without extension')
    parser.add_argument('--quantiles', type=_parse_quantiles, help='Three probabilities that sum to 1, e.g. [0.4, 0.4, 0.2]')
    parser.add_argument('--real-values', action='store_true', help='Accepted for CLI compatibility; ignored by this script')

    args = parser.parse_args()

    try:
        plot_feature_tertile_histograms(
            args.summary_csv,
            args.output,
            args.output_name,
            args.compare_summary_csv,
            args.real_values,
            args.quantiles,
        )
    except Exception as e:
        print(f'✗ Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()