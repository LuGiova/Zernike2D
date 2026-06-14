#!/usr/bin/env python3
"""Plot 2D scatter maps of a complementary plane CSV."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from docs import build_cli_plot_complementary_plane


np.seterr(divide='ignore', invalid='ignore')


def pick_distance_columns(df):
    smoothed_cols = {'smoothed_physical', 'smoothed_zernike'}
    raw_cols = {'physical_distance', 'zernike_distance'}

    # If any smoothed column exists, require the complete smoothed pair and use only that.
    if smoothed_cols.intersection(df.columns):
        if smoothed_cols.issubset(df.columns):
            return 'smoothed_physical', 'smoothed_zernike', True
        raise ValueError(
            'Input CSV has partial smoothed data: both smoothed_physical and '
            'smoothed_zernike are required'
        )

    if raw_cols.issubset(df.columns):
        return 'physical_distance', 'zernike_distance', False

    raise ValueError(
        'Input CSV must contain either physical_distance,zernike_distance or '
        'smoothed_physical,smoothed_zernike'
    )


def plot_plane_subplots(df_plane, phys_col, zernike_col, output_file, use_smoothed=False, cmap='viridis'):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sc0 = axes[0].scatter(
        df_plane['plane_u'],
        df_plane['plane_v'],
        c=df_plane[phys_col],
        cmap=cmap,
        s=18,
        alpha=0.9,
        edgecolors='none',
    )
    axes[0].set_title('Complementary plane colored by physical distance')
    axes[0].set_xlabel('Plane coordinate u')
    axes[0].set_ylabel('Plane coordinate v')
    axes[0].grid(True, alpha=0.25)

    sc1 = axes[1].scatter(
        df_plane['plane_u'],
        df_plane['plane_v'],
        c=df_plane[zernike_col],
        cmap=cmap,
        s=18,
        alpha=0.9,
        edgecolors='none',
    )
    axes[1].set_title('Complementary plane colored by Zernike distance')
    axes[1].set_xlabel('Plane coordinate u')
    axes[1].set_ylabel('Plane coordinate v')
    axes[1].grid(True, alpha=0.25)

    umin, umax = df_plane['plane_u'].min(), df_plane['plane_u'].max()
    vmin, vmax = df_plane['plane_v'].min(), df_plane['plane_v'].max()
    if not np.isclose(umax - umin, 0.0) and not np.isclose(vmax - vmin, 0.0):
        for axis in axes:
            axis.set_aspect('equal', adjustable='box')

    cbar0 = fig.colorbar(sc0, ax=axes[0])
    cbar0.set_label('Physical distance (A)')
    cbar1 = fig.colorbar(sc1, ax=axes[1])
    cbar1.set_label('Zernike distance')

    suffix = ' (smoothed data)' if use_smoothed else ''
    fig.suptitle(f'Complementary plane: physical vs Zernike{suffix}')
    fig.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    args = build_cli_plot_complementary_plane()

    input_csv = Path(args.input)
    output_dir = Path(args.output)

    if not input_csv.exists():
        print(f'Input file not found: {input_csv}', file=sys.stderr)
        sys.exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required_coords = {'plane_u', 'plane_v'}
    if not required_coords.issubset(df.columns):
        print('Input CSV must contain columns: plane_u,plane_v', file=sys.stderr)
        sys.exit(2)

    try:
        phys_col, zern_col, use_smoothed = pick_distance_columns(df)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    output_png = output_dir / f'{input_csv.stem}.png'
    plot_plane_subplots(
        df,
        phys_col,
        zern_col,
        output_png,
        use_smoothed=use_smoothed,
        cmap='viridis',
    )

    print('Figure saved to', output_png)


if __name__ == '__main__':
    main()
