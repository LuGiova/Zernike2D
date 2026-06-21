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
    ('gyration_radius', 'Gyration Radius'),
    ('flatness', 'Flatness'),
    ('roughness', 'Roughness'),
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


def _plot_single_hist(ax, values, title, xlabel, color='#4c78a8', alpha=0.6):
    bins = _hist_bins(values)
    ax.hist(values, bins=bins, color=color, alpha=alpha, edgecolor='black', linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.25, linestyle='--')


def _plot_overlay_hist(ax, values_weighted, values_normal, title, xlabel, show_legend=True):
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
        ax.legend(title='Summary type', fontsize=9, loc='best')
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

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
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
        'Gyration Radius',
        color='#2ca02c',
        alpha=0.70,
    )
    _plot_single_hist(
        axes[0, 1],
        flatness_values,
        'Flatness',
        'Flatness',
        color='#9467bd',
        alpha=0.70,
    )

    # Second row: roughness (single histogram, normal only) and scalar_prod (weighted vs normal).
    roughness_values = _finite_series(df, 'roughness', summary_type='normal')
    _plot_single_hist(
        axes[1, 0],
        roughness_values,
        'Roughness',
        'Roughness',
        color='#d62728',
        alpha=0.70,
    )

    scalar_weighted = _finite_series(df, 'scalar_prod', summary_type='weighted')
    scalar_normal = _finite_series(df, 'scalar_prod', summary_type='normal')
    _plot_overlay_hist(
        axes[1, 1],
        scalar_weighted,
        scalar_normal,
        'Scalar Product',
        'Scalar Product',
        show_legend=has_both_types,
    )

    # Third row: scalar_prod_uncertainty weighted vs normal; left panel for readability, right panel hidden.
    scalar_unc_weighted = _finite_series(df, 'scalar_prod_uncertainty', summary_type='weighted')
    scalar_unc_normal = _finite_series(df, 'scalar_prod_uncertainty', summary_type='normal')
    _plot_overlay_hist(
        axes[2, 0],
        scalar_unc_weighted,
        scalar_unc_normal,
        'Scalar Product Uncertainty',
        'Scalar Product Uncertainty',
        show_legend=has_both_types,
    )
    axes[2, 1].axis('off')

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
