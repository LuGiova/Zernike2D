#!/usr/bin/env python3
"""Plot per-ring histograms for physical and zernike distances from a summary CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RING_COUNT = 10


def _ring_ids(columns: list[str], prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_(?:mean|uncertainty)$")
    ring_ids = sorted({int(match.group(1)) for column in columns if (match := pattern.match(column)) is not None})
    if not ring_ids:
        raise ValueError(f'No columns found for prefix {prefix!r}')
    return ring_ids


def _finite_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.array([], dtype=float)

    values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _hist_bins(*series: np.ndarray) -> np.ndarray | int:
    values = [series_values for series_values in series if len(series_values) > 0]
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


def _output_file(output_dir: Path, summary_path: Path, kind: str) -> Path:
    stem = summary_path.stem
    if stem.endswith('_summary'):
        stem = stem[:-len('_summary')]
    return output_dir / f'{stem}_{kind}_histograms.pdf'


def _plot_histograms(df: pd.DataFrame, ring_ids: list[int], prefix: str, title: str, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for index, ring_id in enumerate(ring_ids[:RING_COUNT]):
        ax = axes[index]
        values = _finite_values(df, f'{prefix}{ring_id}_mean')
        bins = _hist_bins(values)
        ax.hist(values, bins=bins, color='#1f77b4', alpha=0.75, edgecolor='black', linewidth=0.8)
        ax.set_title(f'Ring {ring_id}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.25, linestyle='--')

    for ax in axes[len(ring_ids[:RING_COUNT]):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_ring_histograms(summary_csv_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    summary_path = Path(summary_csv_path)
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary file not found: {summary_path}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    ring_ids = _ring_ids(df.columns.tolist(), 'physical_ring')

    physical_output = _output_file(output_path, summary_path, 'physical')
    zernike_output = _output_file(output_path, summary_path, 'zernike')

    print(f'Loaded summary from {summary_path}')
    print(f'  Shape: {df.shape}')
    print(f'  Ring count: {len(ring_ids)}')

    _plot_histograms(df, ring_ids, 'physical_ring', 'Physical ring histograms', physical_output)
    _plot_histograms(df, ring_ids, 'zernike_ring', 'Zernike ring histograms', zernike_output)

    print(f'Physical plot saved to {physical_output}')
    print(f'Zernike plot saved to {zernike_output}')
    return physical_output, zernike_output


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot per-ring histograms from a summary CSV file.')
    parser.add_argument('summary_csv', help='Path to the input summary CSV')
    parser.add_argument('output_dir', help='Directory where the output figures will be saved')

    args = parser.parse_args()

    try:
        plot_ring_histograms(args.summary_csv, args.output_dir)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()