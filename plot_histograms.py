#!/usr/bin/env python3
"""Plot global metric histograms from a complementary-plane summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ('gyration_radius', 'Gyration Radius (Å)'),
    ('flatness', 'Flatness'),
    ('roughness', 'Roughness (Å)'),
    ('scalar_prod', 'Scalar Product'),
    ('scalar_prod_uncertainty', 'Scalar Product Uncertainty'),
]

COLORS = {
    'weighted': '#1f77b4',
    'normal': '#ff7f0e',
}

ALPHAS = {
    'weighted': 0.55,
    'normal': 0.45,
}


def _pairwise_correlation(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    summary_type: str | None = None,
) -> tuple[float | None, float | None]:
    if metric_a not in df.columns or metric_b not in df.columns:
        return None, None

    subset = df
    if summary_type is not None and 'summary_type' in df.columns:
        subset = subset[subset['summary_type'] == summary_type]

    pair_df = pd.DataFrame(
        {
            metric_a: pd.to_numeric(subset[metric_a], errors='coerce'),
            metric_b: pd.to_numeric(subset[metric_b], errors='coerce'),
        }
    ).dropna()

    if len(pair_df) < 2:
        return None, None

    pearson = pair_df[metric_a].corr(pair_df[metric_b], method='pearson')
    spearman = pair_df[metric_a].corr(pair_df[metric_b], method='spearman')

    if pd.isna(pearson):
        pearson = None
    if pd.isna(spearman):
        spearman = None
    return pearson, spearman


def _correlation_label(
    df: pd.DataFrame,
    metric: str,
    other_metrics: list[str],
    summary_type: str | None = 'normal',
    title_suffix: str | None = None,
) -> str:
    lines = []
    if title_suffix:
        lines.append(title_suffix)

    for other in other_metrics:
        pearson, spearman = _pairwise_correlation(df, metric, other, summary_type=summary_type)
        pretty_other = other.replace('_', ' ')
        if pearson is None or spearman is None:
            lines.append(f'{pretty_other}: P=n/a, S=n/a')
        else:
            lines.append(f'{pretty_other}: P={pearson:.2f}, S={spearman:.2f}')

    return 'Corr. with others\n' + '\n'.join(lines)


def _add_correlation_box(ax, label: str) -> None:
    ax.text(
        0.98,
        0.98,
        label,
        transform=ax.transAxes,
        va='top',
        ha='right',
        fontsize=8,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '#666666'},
    )


def _scalar_correlation_label(df: pd.DataFrame, other_metrics: list[str], has_both_types: bool) -> str:
    lines = ['Corr. with others']
    summary_types = ['normal', 'weighted'] if has_both_types else [None]

    for summary_type in summary_types:
        for other in other_metrics:
            pearson, spearman = _pairwise_correlation(df, 'scalar_prod', other, summary_type=summary_type)
            pretty_other = other.replace('_', ' ')
            type_prefix = ''
            if has_both_types and summary_type is not None:
                type_prefix = f'[{summary_type}] '

            if pearson is None or spearman is None:
                lines.append(f'{type_prefix}{pretty_other}: P=n/a, S=n/a')
            else:
                lines.append(f'{type_prefix}{pretty_other}: P={pearson:.2f}, S={spearman:.2f}')

    return '\n'.join(lines)


def _normalize_stem(summary_path: Path) -> str:
    return summary_path.stem.replace('_summary', '')


def _resolve_output_path(summary_path: Path, output_path: str | Path | None = None, output_name: str | None = None) -> Path:
    default_stem = output_name or f'{_normalize_stem(summary_path)}_histograms'
    if output_path is None:
        return summary_path.parent / f'{default_stem}.pdf'

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / f'{default_stem}.pdf'
    if output_path.suffix:
        return output_path.with_name(f'{default_stem}{output_path.suffix}') if output_name else output_path
    return output_path / f'{default_stem}.pdf'


def _finite_series(df: pd.DataFrame, column: str, summary_type: str | None = None) -> np.ndarray:
    if column not in df.columns:
        return np.array([], dtype=float)

    subset = df
    if summary_type is not None and 'summary_type' in df.columns:
        subset = subset[subset['summary_type'] == summary_type]

    values = pd.to_numeric(subset[column], errors='coerce').to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _hist_bins(*series: np.ndarray) -> np.ndarray | int:
    values = [series_values for series_values in series if len(series_values) > 0]
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


def _label_with_angstrom(label: str) -> str:
    if label in {'Gyration Radius', 'Roughness', 'Radius'}:
        return f'{label} (Å)'
    return label


def _plot_single_hist(ax, values, title, xlabel, color='#4c78a8', alpha=0.6):
    bins = _hist_bins(values)
    ax.hist(values, bins=bins, color=color, alpha=alpha, edgecolor='black', linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.25, linestyle='--')


def _plot_overlay_hist(ax, values_weighted, values_normal, title, xlabel, show_legend=True, legend_loc='best'):
    bins = _hist_bins(values_weighted, values_normal)
    ax.hist(
        values_weighted,
        bins=bins,
        color=COLORS['weighted'],
        alpha=ALPHAS['weighted'],
        edgecolor='black',
        linewidth=0.8,
        label='weighted',
    )
    ax.hist(
        values_normal,
        bins=bins,
        color=COLORS['normal'],
        alpha=ALPHAS['normal'],
        edgecolor='black',
        linewidth=0.8,
        label='normal',
    )
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    if show_legend:
        ax.legend(title='Summary type', fontsize=9, loc=legend_loc)
    ax.grid(axis='y', alpha=0.25, linestyle='--')


def plot_summary_histograms(summary_csv_path: str | Path, output_path: str | Path | None = None, output_name: str | None = None) -> Path:
    summary_path = Path(summary_csv_path)
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary file not found: {summary_path}')

    df = pd.read_csv(summary_path)
    if 'summary_type' not in df.columns:
        raise ValueError('summary_type column not found in summary CSV')

    print(f'Loaded summary from {summary_path}')
    print(f'  Shape: {df.shape}')
    print(f"  Summary types: {df['summary_type'].dropna().unique().tolist()}")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.reshape(3, 2)
    
    # Check if both weighted and normal data are present
    has_both_types = 'weighted' in df['summary_type'].values and 'normal' in df['summary_type'].values

    # First row: single histograms from normal rows only.
    gyration_values = _finite_series(df, 'gyration_radius', summary_type='normal')
    flatness_values = _finite_series(df, 'flatness', summary_type='normal')
    _plot_single_hist(
        axes[0, 0],
        gyration_values,
        'Gyration Radius',
        _label_with_angstrom('Gyration Radius'),
        color='#2ca02c',
        alpha=0.70,
    )
    # correlation boxes removed in favor of a correlation matrix subplot
    _plot_single_hist(
        axes[0, 1],
        flatness_values,
        'Flatness',
        'Flatness',
        color='#9467bd',
        alpha=0.70,
    )
    # correlation boxes removed in favor of a correlation matrix subplot

    # Second row: roughness (single histogram, normal only) and scalar_prod (weighted vs normal).
    roughness_values = _finite_series(df, 'roughness', summary_type='normal')
    _plot_single_hist(
        axes[1, 0],
        roughness_values,
        'Roughness',
        _label_with_angstrom('Roughness'),
        color='#d62728',
        alpha=0.70,
    )
    # correlation boxes removed in favor of a correlation matrix subplot

    scalar_weighted = _finite_series(df, 'scalar_prod', summary_type='weighted')
    scalar_normal = _finite_series(df, 'scalar_prod', summary_type='normal')
    _plot_overlay_hist(
        axes[1, 1],
        scalar_weighted,
        scalar_normal,
        'Scalar Product',
        'Scalar Product',
        show_legend=has_both_types,
        legend_loc='upper left',
    )
    # scalar correlation box removed; will show matrix instead

    # Third row: radius histogram (normal only). Leave the last panel empty.
    radius_values = _finite_series(df, 'radius', summary_type='normal')
    _plot_single_hist(
        axes[2, 0],
        radius_values,
        'Radius',
        _label_with_angstrom('Radius'),
        color='#17becf',
        alpha=0.70,
    )
    # Build and plot a correlation matrix instead of small correlation boxes.
    # Metrics: gyration_radius, flatness, roughness, scalar_prod (split by summary_type if both present), scalar_prod_uncertainty
    cols = []
    base_metrics = ['gyration_radius', 'flatness', 'roughness']
    for m in base_metrics:
        if m in df.columns:
            cols.append(m)

    # handle scalar_prod splitting when both summary types exist
    if 'scalar_prod' in df.columns:
        if has_both_types:
            cols.append('scalar_prod_weighted')
            cols.append('scalar_prod_normal')
        else:
            cols.append('scalar_prod')

    # include radius instead of scalar_prod_uncertainty
    if 'radius' in df.columns:
        cols.append('radius')

    # build aligned series dataframe (keep original index so pairwise corr uses pairwise dropna)
    corr_data = {}
    for col in cols:
        if col == 'scalar_prod_weighted':
            s = pd.Series(pd.NA, index=df.index, dtype='float64')
            mask = df['summary_type'] == 'weighted'
            s.loc[mask] = pd.to_numeric(df.loc[mask, 'scalar_prod'], errors='coerce')
            corr_data[col] = s
        elif col == 'scalar_prod_normal':
            s = pd.Series(pd.NA, index=df.index, dtype='float64')
            mask = df['summary_type'] == 'normal'
            s.loc[mask] = pd.to_numeric(df.loc[mask, 'scalar_prod'], errors='coerce')
            corr_data[col] = s
        else:
            corr_data[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series([pd.NA] * len(df))

    corr_df = pd.DataFrame(corr_data)
    corr_pearson = corr_df.corr(method='pearson')
    corr_spearman = corr_df.corr(method='spearman')

    # create two small axes inside the area reserved for axes[2,1]
    area = axes[2, 1].get_position()
    # remove placeholder axis
    axes[2, 1].remove()

    # compute matrix widths and center the two matrices horizontally inside the reserved area
    matrix_w = area.width * 0.42
    gap = area.width * 0.16
    total_w = matrix_w * 2 + gap
    x_center = area.x0 + (area.width - total_w) / 2.0
    # small positive x_shift moves both matrices right; increase y_shift to move them down
    x_shift = area.width * 0.03
    y_shift = area.height * 0.14
    left_bbox = [x_center + x_shift, area.y0 - y_shift, matrix_w, area.height]
    right_bbox = [x_center + x_shift + matrix_w + gap, area.y0 - y_shift, matrix_w, area.height]

    ax_p = fig.add_axes(left_bbox)
    im_p = ax_p.imshow(corr_pearson.values, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='auto')
    ax_p.set_xticks(range(len(corr_pearson.columns)))
    ax_p.set_yticks(range(len(corr_pearson.index)))
    # make tick labels smaller to avoid overlap; matrix cell numbers stay readable
    ax_p.set_xticklabels(corr_pearson.columns, rotation=45, ha='right', fontsize=7)
    ax_p.set_yticklabels(corr_pearson.index, fontsize=7)
    ax_p.set_title('Pearson', fontsize=9)
    for i in range(corr_pearson.shape[0]):
        for j in range(corr_pearson.shape[1]):
            val = corr_pearson.iat[i, j]
            txt = 'n/a' if pd.isna(val) else f"{val:.2f}"
            ax_p.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)
    # no colorbar (keep colored cells only)

    ax_s = fig.add_axes(right_bbox)
    im_s = ax_s.imshow(corr_spearman.values, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='auto')
    ax_s.set_xticks(range(len(corr_spearman.columns)))
    ax_s.set_yticks(range(len(corr_spearman.index)))
    ax_s.set_xticklabels(corr_spearman.columns, rotation=45, ha='right', fontsize=7)
    ax_s.set_yticklabels(corr_spearman.index, fontsize=7)
    ax_s.set_title('Spearman', fontsize=9)
    for i in range(corr_spearman.shape[0]):
        for j in range(corr_spearman.shape[1]):
            val = corr_spearman.iat[i, j]
            txt = 'n/a' if pd.isna(val) else f"{val:.2f}"
            ax_s.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)
    # no colorbar (keep colored cells only)

    fig.suptitle(f'Global Metric Histograms ({_normalize_stem(summary_path)})', fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout()

    resolved_output = _resolve_output_path(summary_path, output_path, output_name)
    fig.savefig(resolved_output, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Plot saved to {resolved_output}')
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot histograms for global summary metrics from a CSV file.')
    parser.add_argument('summary_csv', help='Path to summary CSV file')
    parser.add_argument('-o', '--output', help='Output path or directory for the figure')
    parser.add_argument('--output-name', help='Output file name stem without extension')

    args = parser.parse_args()

    try:
        plot_summary_histograms(args.summary_csv, args.output, args.output_name)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
