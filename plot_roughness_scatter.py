#!/usr/bin/env python3
"""Scatter plot of global roughness values from two summary CSV files.

The script aligns complexes by `complex_name`, compares the global `roughness`
column from the first file against the global `roughness` column from the
second file, and draws a scatter split into low, medium, and large tertiles
using the `gyration_radius` values from the first CSV.

For each tertile subplot, the script finds the largest `m > 1` for which the
two wedges around the bisector inside the band between `y = m x` and
`y = (1/m) x` remain approximately balanced, then colors the three zones.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Scatter plot of global roughness values from two summary CSV files.'
    )
    parser.add_argument('first_csv', type=Path, help='First summary CSV file')
    parser.add_argument('second_csv', type=Path, help='Second summary CSV file')
    parser.add_argument('output_plot', type=Path, help='Output plot path, e.g. ./output_files/roughness_scatter.pdf')
    parser.add_argument(
        '--complex-col',
        default='complex_name',
        help='Name of the complex ID column. Default: complex_name.',
    )
    parser.add_argument(
        '--decoy-suffix-regex',
        default=r'-\d+$',
        help=(
            'Regex removed from complex names in the second CSV before matching. '
            "Default: '-\\d+$', removing suffixes like -1, -23, -004."
        ),
    )
    parser.add_argument(
        '--title',
        default='Global roughness comparison',
        help='Plot title. Default: Global roughness comparison.',
    )
    parser.add_argument(
        '--zone-tolerance',
        type=float,
        default=0.10,
        help='Relative tolerance used to balance the two wedges around the bisector. Default: 0.10.',
    )
    return parser


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        na_values=['', ' ', 'NA', 'NaN', 'nan', 'None', 'null', 'NULL'],
        keep_default_na=True,
    )
    df.columns = df.columns.str.strip()
    return df


def _normalize_second_complex_name(value: object, suffix_regex: str) -> str:
    text = '' if pd.isna(value) else str(value).strip()
    return re.sub(suffix_regex, '', text)


def _validate_inputs(df: pd.DataFrame, label: str, complex_col: str, require_gyration_radius: bool = False) -> None:
    if complex_col not in df.columns:
        raise ValueError(f"Column {complex_col!r} not found in {label} CSV")
    if 'roughness' not in df.columns:
        raise ValueError(f"Column 'roughness' not found in {label} CSV")
    if require_gyration_radius and 'gyration_radius' not in df.columns:
        raise ValueError(f"Column 'gyration_radius' not found in {label} CSV")


def _deduplicate(df: pd.DataFrame, complex_col: str, label: str) -> pd.DataFrame:
    duplicated = df[complex_col].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, complex_col].astype(str).unique()[:10]
        raise ValueError(
            f'Duplicate complex IDs found in {label} after matching. Examples: {examples}. '
            'If you really have multiple rows per complex, aggregate them before plotting.'
        )
    return df


def _prepare_matched_pairs(
    first_df: pd.DataFrame,
    second_df: pd.DataFrame,
    complex_col: str,
    suffix_regex: str,
) -> pd.DataFrame:
    first = first_df.copy()
    second = second_df.copy()

    first['_match_key'] = first[complex_col].astype(str).str.strip()
    second['_match_key'] = second[complex_col].map(lambda value: _normalize_second_complex_name(value, suffix_regex))

    first = _deduplicate(first, '_match_key', 'first')
    second = _deduplicate(second, '_match_key', 'second')

    merged = first.merge(
        second,
        on='_match_key',
        how='inner',
        suffixes=('_first', '_second'),
    )

    if merged.empty:
        raise ValueError('No common complexes found after matching the two CSV files')

    merged['roughness_first'] = pd.to_numeric(merged['roughness_first'], errors='coerce')
    merged['roughness_second'] = pd.to_numeric(merged['roughness_second'], errors='coerce')
    merged = merged.dropna(subset=['roughness_first', 'roughness_second'])

    if merged.empty:
        raise ValueError('Matched complexes exist, but no valid roughness values were found')

    return merged


def _axis_limits(values_x: np.ndarray, values_y: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([values_x, values_y])
    data_min = float(np.min(combined))
    data_max = float(np.max(combined))
    if np.isclose(data_min, data_max):
        padding = max(abs(data_min) * 0.05, 1.0)
        return min(0.0, data_min - padding), data_max + padding

    padding = (data_max - data_min) * 0.05
    return min(0.0, data_min - padding), data_max + padding


def _tertile_thresholds(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        raise ValueError('No valid gyration_radius values found to compute tertiles')

    low_threshold = float(np.nanquantile(finite_values, 1.0 / 3.0))
    high_threshold = float(np.nanquantile(finite_values, 2.0 / 3.0))
    return low_threshold, high_threshold


def _tertile_label(value: float, low_threshold: float, high_threshold: float) -> str:
    if value <= low_threshold:
        return 'low'
    if value <= high_threshold:
        return 'medium'
    return 'large'


def _balanced_wedge_counts(x_values: np.ndarray, y_values: np.ndarray, m_value: float) -> tuple[int, int, int, int, int]:
    valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
    if not np.any(valid):
        return 0, 0, 0, 0, 0

    ratios = y_values[valid] / x_values[valid]
    upper_wedge = int(np.sum((ratios > 1.0) & (ratios <= m_value)))
    lower_wedge = int(np.sum((ratios >= 1.0 / m_value) & (ratios < 1.0)))
    upper_zone = int(np.sum(ratios > m_value))
    lower_zone = int(np.sum(ratios < 1.0 / m_value))
    equal_zone = int(np.sum((ratios >= 1.0 / m_value) & (ratios <= m_value)))
    return upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone


def _select_max_balanced_m(x_values: np.ndarray, y_values: np.ndarray, tolerance: float) -> tuple[float, int, int, int, int, int, float]:
    valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
    if np.sum(valid) < 2:
        raise ValueError('Not enough valid positive points to select m')

    ratios = y_values[valid] / x_values[valid]
    positive_ratios = ratios[ratios > 0]
    if len(positive_ratios) == 0:
        raise ValueError('No positive roughness ratios found to select m')

    min_ratio = float(np.min(positive_ratios))
    max_ratio = float(np.max(positive_ratios))
    upper_bound = max(2.0, max(max_ratio, 1.0 / max(min_ratio, 1e-6)))
    upper_bound = min(upper_bound, 100.0)

    candidate_ms = np.geomspace(1.0001, upper_bound, num=2000)
    best_valid = None
    best_valid_metric = np.inf
    best_valid_counts = (0, 0, 0, 0, 0)
    best_any = None
    best_any_metric = np.inf
    best_any_counts = (0, 0, 0, 0, 0)

    for m_value in candidate_ms:
        upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone = _balanced_wedge_counts(x_values, y_values, float(m_value))
        total_wedges = upper_wedge + lower_wedge
        if total_wedges == 0:
            continue

        metric = abs(upper_wedge - lower_wedge) / float(total_wedges)
        if metric < best_any_metric or (np.isclose(metric, best_any_metric) and (best_any is None or m_value > best_any)):
            best_any = float(m_value)
            best_any_metric = metric
            best_any_counts = (upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone)

        if metric <= tolerance:
            best_valid = float(m_value)
            best_valid_metric = metric
            best_valid_counts = (upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone)

    if best_valid is not None:
        upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone = best_valid_counts
        return best_valid, upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone, best_valid_metric

    if best_any is not None:
        upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone = best_any_counts
        return best_any, upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone, best_any_metric

    raise ValueError('Unable to determine a balanced m for the selected tertile')


def _zone_percentages(top_mask: np.ndarray, equal_mask: np.ndarray, bottom_mask: np.ndarray) -> tuple[float, float, float]:
    total = int(np.sum(top_mask | equal_mask | bottom_mask))
    if total == 0:
        return 0.0, 0.0, 0.0

    top_pct = float(np.sum(top_mask) / total * 100.0)
    equal_pct = float(np.sum(equal_mask) / total * 100.0)
    bottom_pct = float(np.sum(bottom_mask) / total * 100.0)
    return top_pct, equal_pct, bottom_pct


def _hist_bins(*series: np.ndarray) -> np.ndarray | int:
    values = [np.asarray(series_values, dtype=float) for series_values in series if len(series_values) > 0]
    if not values:
        return 10

    combined = np.concatenate(values)
    if len(combined) < 2:
        return 10

    data_min = float(np.min(combined))
    data_max = float(np.max(combined))
    if np.isclose(data_min, data_max):
        return 10

    return np.histogram_bin_edges(combined, bins='auto')


def _plot_panel_with_marginals(
    fig: plt.Figure,
    outer_spec,
    x_masked: np.ndarray,
    y_masked: np.ndarray,
    first_label: str,
    second_label: str,
    x_label: str,
    y_label: str,
    tertile_name: str,
    m_value: float,
    zone_tolerance: float,
    lower: float,
    upper: float,
    zone_colors: dict[str, str],
    top_mask: np.ndarray,
    equal_mask: np.ndarray,
    bottom_mask: np.ndarray,
    top_pct: float,
    equal_pct: float,
    bottom_pct: float,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    inner = outer_spec.subgridspec(2, 2, width_ratios=[1.0, 4.0], height_ratios=[4.0, 1.0], wspace=0.05, hspace=0.05)
    ax_histy = fig.add_subplot(inner[0, 0])
    ax_scatter = fig.add_subplot(inner[0, 1], sharey=ax_histy)
    ax_histx = fig.add_subplot(inner[1, 1], sharex=ax_scatter)
    ax_blank = fig.add_subplot(inner[1, 0])
    ax_blank.axis('off')

    scatter_kwargs = dict(s=28, alpha=0.85, edgecolors='black', linewidths=0.4)
    ax_scatter.scatter(x_masked[top_mask], y_masked[top_mask], color=zone_colors['upper'], label=f'{second_label} > {first_label}', zorder=3, **scatter_kwargs)
    ax_scatter.scatter(x_masked[equal_mask], y_masked[equal_mask], color=zone_colors['equal'], label='equal', zorder=3, **scatter_kwargs)
    ax_scatter.scatter(x_masked[bottom_mask], y_masked[bottom_mask], color=zone_colors['lower'], label=f'{first_label} > {second_label}', zorder=3, **scatter_kwargs)

    x_fill = np.linspace(lower, upper, 400)
    upper_line = m_value * x_fill
    lower_line = x_fill / m_value
    ax_scatter.fill_between(x_fill, upper_line, upper, color=zone_colors['upper'], alpha=0.12, zorder=0)
    ax_scatter.fill_between(x_fill, lower_line, upper_line, color=zone_colors['equal'], alpha=0.16, zorder=0)
    ax_scatter.fill_between(x_fill, lower, lower_line, color=zone_colors['lower'], alpha=0.12, zorder=0)
    ax_scatter.plot([lower, upper], [m_value * lower, m_value * upper], linestyle='--', color='#d62728', linewidth=1.6)
    ax_scatter.plot([lower, upper], [lower / m_value, upper / m_value], linestyle='-.', color='#1f77b4', linewidth=1.6)
    ax_scatter.set_title(f'{tertile_name.capitalize()} gyration_radius  (m = {m_value:.2f})')
    ax_scatter.set_xlabel('')
    ax_scatter.set_ylabel('')
    ax_scatter.grid(alpha=0.25, linestyle='--')

    x_bins = _hist_bins(x_masked)
    y_bins = _hist_bins(y_masked)
    ax_histx.hist(x_masked[np.isfinite(x_masked)], bins=x_bins, color='#666666', alpha=0.75, edgecolor='black', linewidth=0.6)
    ax_histy.hist(y_masked[np.isfinite(y_masked)], bins=y_bins, orientation='horizontal', color='#666666', alpha=0.75, edgecolor='black', linewidth=0.6)

    ax_histx.set_xlabel(x_label)
    ax_histx.set_ylabel('Count')
    ax_histy.set_xlabel('Count')
    ax_histy.set_ylabel(y_label)
    ax_histx.grid(axis='y', alpha=0.25, linestyle='--')
    ax_histy.grid(axis='x', alpha=0.25, linestyle='--')

    plt.setp(ax_histy.get_xticklabels(), visible=False)
    plt.setp(ax_histx.get_yticklabels(), visible=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    ax_histx.tick_params(axis='x', labelbottom=False)

    legend_handles = [
        Patch(facecolor=zone_colors['upper'], edgecolor='black', alpha=0.25, label=f'{second_label} > {first_label} ({top_pct:.1f}%)'),
        Patch(facecolor=zone_colors['equal'], edgecolor='black', alpha=0.25, label=f'equal ({equal_pct:.1f}%)'),
        Patch(facecolor=zone_colors['lower'], edgecolor='black', alpha=0.25, label=f'{first_label} > {second_label} ({bottom_pct:.1f}%)'),
        Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1.6, label=f'y = {m_value:.2f}x'),
        Line2D([0], [0], color='#1f77b4', linestyle='-.', linewidth=1.6, label=f'y = (1/{m_value:.2f})x'),
    ]
    ax_scatter.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=8,
        title=f'{first_label} vs {second_label}\nbalanced diff ≤ {zone_tolerance:.0%}',
    )

    return ax_scatter, ax_histx, ax_histy


def plot_roughness_scatter(
    first_csv: Path,
    second_csv: Path,
    output_plot: Path,
    complex_col: str = 'complex_name',
    suffix_regex: str = r'-\d+$',
    title: str = 'Global roughness comparison',
    zone_tolerance: float = 0.10,
) -> Path:
    first_df = _read_csv(first_csv)
    second_df = _read_csv(second_csv)

    _validate_inputs(first_df, 'first', complex_col, require_gyration_radius=True)
    _validate_inputs(second_df, 'second', complex_col)

    matched = _prepare_matched_pairs(first_df, second_df, complex_col, suffix_regex)

    x = matched['roughness_first'].to_numpy(dtype=float)
    y = matched['roughness_second'].to_numpy(dtype=float)
    gyration_radius = pd.to_numeric(matched['gyration_radius_first'], errors='coerce').to_numpy(dtype=float)
    low_threshold, high_threshold = _tertile_thresholds(gyration_radius)
    tertiles = np.array([_tertile_label(value, low_threshold, high_threshold) for value in gyration_radius], dtype=object)
    first_label = first_csv.stem
    second_label = second_csv.stem

    output_plot.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(21, 7.5), constrained_layout=True)
    outer = fig.add_gridspec(1, 3, wspace=0.25)
    tertile_colors = {
        'low': '#8ecae6',
        'medium': '#b7e4c7',
        'large': '#ffb703',
    }
    zone_colors = {
        'upper': '#d9534f',
        'equal': '#8ecae6',
        'lower': '#4c78a8',
    }

    lower, upper = _axis_limits(x, y)
    scatter_axes = []
    for outer_spec, tertile_name in zip(outer, ['low', 'medium', 'large']):
        mask = tertiles == tertile_name
        x_masked = x[mask]
        y_masked = y[mask]
        m_value, upper_wedge, lower_wedge, upper_zone, lower_zone, equal_zone, metric = _select_max_balanced_m(
            x_masked,
            y_masked,
            zone_tolerance,
        )

        valid = np.isfinite(x_masked) & np.isfinite(y_masked) & (x_masked > 0) & (y_masked > 0)
        ratios = np.full_like(x_masked, np.nan, dtype=float)
        ratios[valid] = y_masked[valid] / x_masked[valid]
        top_mask = valid & (ratios > m_value)
        bottom_mask = valid & (ratios < 1.0 / m_value)
        equal_mask = valid & ~(top_mask | bottom_mask)
        top_pct, equal_pct, bottom_pct = _zone_percentages(top_mask, equal_mask, bottom_mask)

        ax_scatter, ax_histx, ax_histy = _plot_panel_with_marginals(
            fig,
            outer_spec,
            x_masked,
            y_masked,
            first_label,
            second_label,
            f'Roughness in {first_csv.name}',
            f'Roughness in {second_csv.name}',
            tertile_name,
            m_value,
            zone_tolerance,
            lower,
            upper,
            zone_colors,
            top_mask,
            equal_mask,
            bottom_mask,
            top_pct,
            equal_pct,
            bottom_pct,
        )
        scatter_axes.append(ax_scatter)

        print(
            f'  {tertile_name}: m={m_value:.4f}, upper_wedge={upper_wedge}, lower_wedge={lower_wedge}, '
            f'upper_zone={upper_zone}, lower_zone={lower_zone}, equal_zone={equal_zone}, '
            f'rel_diff={metric:.3f}'
        )

    for ax in scatter_axes:
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(title)
    fig.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_plot


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_path = plot_roughness_scatter(
        args.first_csv,
        args.second_csv,
        args.output_plot,
        complex_col=args.complex_col,
        suffix_regex=args.decoy_suffix_regex,
        title=args.title,
        zone_tolerance=args.zone_tolerance,
    )
    print(f'Wrote plot to {output_path}')


if __name__ == '__main__':
    main()