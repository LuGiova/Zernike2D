#!/usr/bin/env python3
"""Analyze correlation between physical and Zernike distances on an interface.

Steps:
1. Spatial smoothing (KDTree rolling mean within radius)
2. Spearman correlation (raw and smoothed)
3. Scatter plots (raw vs smoothed)

Usage:
  python analyze_interface_correlations.py -i input.csv -o out.png -r 6.0

"""
from pathlib import Path
import sys
from docs import build_cli_interface_correlation

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree
from scipy.stats import spearmanr
from pykrige.ok import OrdinaryKriging
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.pyplot as plt


def smooth_with_kdtree(df, radius=6.0, sigma=None, d_penalty=None):
    # Option 2: distance-combined weighting (requires original 3D coords)
    if not {'x1', 'y1', 'z1', 'x2', 'y2', 'z2'}.issubset(df.columns):
        raise ValueError('Input CSV must contain original coords: x1,y1,z1,x2,y2,z2 for 3D-based smoothing')

    coords1 = df[['x1', 'y1', 'z1']].to_numpy(dtype=float)
    coords2 = df[['x2', 'y2', 'z2']].to_numpy(dtype=float)

    tree1 = KDTree(coords1)
    tree2 = KDTree(coords2)
    neigh1 = tree1.query_ball_point(coords1, r=radius)
    neigh2 = tree2.query_ball_point(coords2, r=radius)

    phys = df['physical_distance'].to_numpy(dtype=float)
    zern = df['zernike_distance'].to_numpy(dtype=float)

    n = len(df)
    smoothed_phys = np.empty(n, dtype=float)
    smoothed_zern = np.empty(n, dtype=float)

    if sigma is None:
        sigma = max(1e-6, radius / 2.0)
    if d_penalty is None:
        d_penalty = radius

    for i in range(n):
        n1 = set(neigh1[i])
        n2 = set(neigh2[i])
        union = n1.union(n2)
        # ensure self included
        union.add(i)

        idxs = np.fromiter(union, dtype=int)

        # compute combined distance D for each candidate
        d1 = np.empty(len(idxs), dtype=float)
        d2 = np.empty(len(idxs), dtype=float)
        for k, j in enumerate(idxs):
            if j in n1:
                d1[k] = np.linalg.norm(coords1[i] - coords1[j])
            else:
                d1[k] = d_penalty
            if j in n2:
                d2[k] = np.linalg.norm(coords2[i] - coords2[j])
            else:
                d2[k] = d_penalty

        D = np.sqrt(d1 * d1 + d2 * d2)
        weights = np.exp(-D / sigma)
        # avoid zero-sum
        if weights.sum() <= 0:
            smoothed_phys[i] = phys[i]
            smoothed_zern[i] = zern[i]
        else:
            smoothed_phys[i] = np.sum(weights * phys[idxs]) / np.sum(weights)
            smoothed_zern[i] = np.sum(weights * zern[idxs]) / np.sum(weights)

    out = df.copy()
    out['smoothed_physical'] = smoothed_phys
    out['smoothed_zernike'] = smoothed_zern
    return out


def load_and_smooth_interface_data(csv_file, radius=6.0):
    df = pd.read_csv(csv_file)
    required_cols = {'physical_distance', 'zernike_distance'}
    if not required_cols.issubset(df.columns):
        raise ValueError('Input CSV must contain columns: physical_distance,zernike_distance')
    return smooth_with_kdtree(df, radius=radius)


def compute_kriging_layer(u, v, z, grid_size=150):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)

    grid_u = np.linspace(u.min() - 2.0, u.max() + 2.0, grid_size)
    grid_v = np.linspace(v.min() - 2.0, v.max() + 2.0, grid_size)

    kriging = OrdinaryKriging(
        u,
        v,
        z,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False,
    )
    z_interp, sigmasq = kriging.execute('grid', grid_u, grid_v)

    z_layer = np.array(np.ma.getdata(z_interp), dtype=float, copy=True)
    sigma_layer = np.array(np.ma.getdata(sigmasq), dtype=float, copy=False)
    return grid_u, grid_v, z_layer, sigma_layer


def variance_to_alpha(sigmasq, alpha_gamma=2.0):
    sigma = np.asarray(sigmasq, dtype=float)
    alpha = np.zeros_like(sigma, dtype=float)

    valid = np.isfinite(sigma)
    if not np.any(valid):
        return alpha

    sigma_valid = sigma[valid]
    sigma_min = sigma_valid.min()
    sigma_max = sigma_valid.max()

    if np.isclose(sigma_max, sigma_min):
        alpha[valid] = 1.0
        return alpha

    normalized = (sigma_valid - sigma_min) / (sigma_max - sigma_min)
    alpha[valid] = np.power(1.0 - normalized, alpha_gamma)
    alpha[~valid] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def truncate_colormap(cmap, minval=0.06, maxval=1.0, n=256):
    """Return a truncated version of a Colormap.

    Parameters:
    - cmap: colormap name or Colormap instance
    - minval: fraction in [0,1] to start the colormap (cuts the darkest colors)
    - maxval: fraction in [0,1] to end the colormap
    - n: number of color samples
    """
    if isinstance(cmap, str):
        base = plt.get_cmap(cmap)
    else:
        base = cmap
    new_colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{base.name}_trunc_{minval:.2f}", new_colors)


def plot_kriging_grid(csv_file, radius=6.0, df=None, output_file=None):
    csv_path = Path(csv_file)
    if df is None:
        df = load_and_smooth_interface_data(csv_path, radius=radius)

    required_cols = {'plane_u', 'plane_v', 'physical_distance', 'zernike_distance', 'smoothed_physical', 'smoothed_zernike'}
    if not required_cols.issubset(df.columns):
        raise ValueError('Topographic maps require plane_u, plane_v, physical_distance, zernike_distance, smoothed_physical, smoothed_zernike columns')

    u = df['plane_u'].to_numpy(dtype=float)
    v = df['plane_v'].to_numpy(dtype=float)

    layers = [
        ('physical_distance', 'Raw data - physical distance', 'inferno', 'Physical distance (Å)'),
        ('zernike_distance', 'Raw data - Zernike distance', 'inferno', 'Zernike distance'),
        ('smoothed_physical', 'Smoothed data - physical distance', 'inferno', 'Smoothed physical distance (Å)'),
        ('smoothed_zernike', 'Smoothed data - Zernike distance', 'inferno', 'Smoothed Zernike distance'),
    ]

    dark_background = '#101010'
    # Use GridSpec with an extra narrow column to the right of each subplot
    # so the alpha strip legend can sit outside the image (not overlapping).
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(dark_background)
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=[1.0, 0.04, 1.0, 0.04], hspace=0.28, wspace=0.18)


    for idx, (column, title, cmap, cbar_label) in enumerate(layers):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col * 2])
        cax = fig.add_subplot(gs[row, col * 2 + 1])
        ax.set_facecolor(dark_background)
        grid_u, grid_v, z_layer, sigma_layer = compute_kriging_layer(
            u,
            v,
            df[column].to_numpy(dtype=float),
        )
        grid_x, grid_y = np.meshgrid(grid_u, grid_v)
        z_layer = np.array(z_layer, dtype=float, copy=True)
        sigma_layer = np.array(sigma_layer, dtype=float, copy=True)

        valid = np.isfinite(z_layer)
        if np.any(valid):
            vmin = np.nanmin(z_layer)
            vmax = np.nanmax(z_layer)
            if np.isclose(vmax, vmin):
                norm = Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)

        cmap_obj = plt.get_cmap(cmap)
        # keep the same truncation for all maps; make Zernike transparency more aggressive
        is_zernike = 'zernike' in column.lower()
        cmap_trunc = truncate_colormap(cmap_obj, minval=0.06)

        # normalize values (no gamma/color remapping here)
        values = np.where(valid, z_layer, norm.vmin)
        values_norm = norm(values)

        rgba = cmap_trunc(values_norm)
        # Make the base map fully opaque where values are valid so contour lines
        # can be drawn on top. Keep color mapping as before.
        rgba[..., 3] = valid.astype(float)

        ax.imshow(
            rgba,
            origin='lower',
            extent=(grid_u.min(), grid_u.max(), grid_v.min(), grid_v.max()),
            interpolation='nearest',
            aspect='equal',
            zorder=1,
        )

        # Foreground "veil": background-colored image whose alpha is the inverse
        # of the variance-based alpha (i.e. opaque at high variance, transparent
        # at low variance). Keep the stronger contrast for Zernike maps by
        # using a larger alpha_gamma for them.
        alpha_gamma = 4.0 if is_zernike else 2.0
        variance_alpha = variance_to_alpha(sigma_layer, alpha_gamma=alpha_gamma)
        veil_alpha = (1.0 - variance_alpha) * valid.astype(float)

        # background color as RGBA
        bg_rgba = matplotlib.colors.to_rgba(dark_background)
        veil_rgba = np.empty(rgba.shape, dtype=float)
        veil_rgba[..., 0] = bg_rgba[0]
        veil_rgba[..., 1] = bg_rgba[1]
        veil_rgba[..., 2] = bg_rgba[2]
        veil_rgba[..., 3] = veil_alpha

        ax.imshow(
            veil_rgba,
            origin='lower',
            extent=(grid_u.min(), grid_u.max(), grid_v.min(), grid_v.max()),
            interpolation='nearest',
            aspect='equal',
            zorder=3,
        )

        # Draw contours under the veil so they appear softened by it.
        contour_handle = ax.contour(grid_x, grid_y, np.ma.masked_invalid(z_layer), levels=10, colors='white', linewidths=0.6, zorder=2)
        label_list = ax.clabel(contour_handle, inline=True, fontsize=8, fmt='%.2f', colors='white')
        # Ensure contour labels are also below the veil
        if label_list is not None:
            try:
                for lbl in label_list:
                    lbl.set_zorder(2)
            except Exception:
                pass
        ax.set_title(title)
        ax.set_xlabel('plane_u')
        ax.set_ylabel('plane_v')
        ax.set_aspect('equal', adjustable='box')
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_trunc)
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, ax=ax)
        colorbar.set_label(cbar_label)
        colorbar.ax.set_facecolor(dark_background)
        colorbar.outline.set_edgecolor('white')
        colorbar.ax.tick_params(colors='white')
        colorbar.ax.yaxis.label.set_color('white')
        plt.setp(colorbar.ax.get_yticklabels(), color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        # --- Alpha strip legend (visual): render into the dedicated GridSpec
        try:
            sigma = np.asarray(sigma_layer, dtype=float)
            valid_sigma = np.isfinite(sigma)
            if np.any(valid_sigma):
                smin = np.nanmin(sigma)
                smax = np.nanmax(sigma)
                if np.isclose(smax, smin):
                    smin = smin - 1e-6
                    smax = smax + 1e-6

                ag = 4.0 if is_zernike else 2.0
                nstrip = 256
                sigma_vals = np.linspace(smin, smax, nstrip)
                var_alpha = variance_to_alpha(sigma_vals, alpha_gamma=ag)
                veil_alpha_strip = (1.0 - var_alpha)

                # Use a single white color for the strip and vary only alpha so
                # it fades to transparency; this makes the transparency visible
                # against the colored map. The strip is narrow (GridSpec column).
                strip = np.ones((nstrip, 10, 4), dtype=float)
                strip[..., 0:3] = 1.0  # white
                strip[..., 3] = veil_alpha_strip[:, None]

                # draw into the preallocated cax from GridSpec
                cax.imshow(strip, origin='lower', aspect='auto', extent=(0, 1, smin, smax))
                cax.set_xlim(0, 1)
                cax.set_xticks([])
                ticks = np.linspace(smin, smax, 3)
                cax.set_yticks(ticks)
                cax.set_yticklabels([f"{t:.2e}" for t in ticks], color='white', fontsize=8)
                cax.tick_params(axis='y', colors='white', labelsize=8, length=3)
                cax.yaxis.set_label_position('right')
                cax.set_ylabel('variance (σ²)', color='white', rotation=270, labelpad=12)
                cax.patch.set_alpha(0.0)
            else:
                cax.axis('off')
        except Exception:
            try:
                cax.axis('off')
            except Exception:
                pass

    fig.suptitle('Topographic maps (Kriging) — transparency modulated by variance (σ²)', color='white')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_file is None:
        output_file = csv_path.with_name(f'{csv_path.stem}_topo_kriging_2x2.png')
    else:
        output_file = Path(output_file)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Topographic figure saved to', output_file)
    return output_file


def compute_and_print_spearman(df):
    x_raw = df['physical_distance'].to_numpy(dtype=float)
    y_raw = df['zernike_distance'].to_numpy(dtype=float)
    rho_raw, p_raw = spearmanr(x_raw, y_raw)

    x_s = df['smoothed_physical'].to_numpy(dtype=float)
    y_s = df['smoothed_zernike'].to_numpy(dtype=float)
    rho_s, p_s = spearmanr(x_s, y_s)

    print('Spearman correlation (raw): rho={:.4f}, p={:.3e}'.format(rho_raw, p_raw))
    print('Spearman correlation (smoothed): rho={:.4f}, p={:.3e}'.format(rho_s, p_s))

    return (rho_raw, p_raw), (rho_s, p_s)


def plot_scatter(df, out_file, rho_raw, rho_s):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(df['physical_distance'], df['zernike_distance'], s=18, alpha=0.6, c='tab:blue')
    axes[0].set_xlabel('physical_distance (Å)')
    axes[0].set_ylabel('zernike_distance')
    axes[0].set_title(f'Raw data — Spearman rho={rho_raw:.3f}')
    axes[0].grid(alpha=0.25)

    axes[1].scatter(df['smoothed_physical'], df['smoothed_zernike'], s=18, alpha=0.6, c='tab:orange')
    axes[1].set_xlabel('smoothed_physical (Å)')
    axes[1].set_ylabel('smoothed_zernike')
    axes[1].set_title(f'Smoothed data — Spearman rho={rho_s:.3f}')
    axes[1].grid(alpha=0.25)

    fig.suptitle('Physical vs Zernike distances (raw and smoothed)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    args = build_cli_interface_correlation()
    path = Path(args.input)
    if not path.exists():
        print('Input file not found:', args.input, file=sys.stderr)
        sys.exit(2)

    df_sm = None
    if getattr(args, 'topo', False):
        try:
            df_sm = load_and_smooth_interface_data(path, radius=args.radius)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)

        topo_output = Path(args.output).with_name(f'{Path(args.output).stem}_topograhy.png')
        plot_kriging_grid(
            path,
            radius=args.radius,
            df=df_sm,
            output_file=topo_output,
        )

    if df_sm is None:
        try:
            df_sm = load_and_smooth_interface_data(path, radius=args.radius)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)

    (rho_raw, p_raw), (rho_s, p_s) = compute_and_print_spearman(df_sm)
    plot_scatter(df_sm, args.output, rho_raw, rho_s)
    print('Figure saved to', args.output)
    if getattr(args, 'save_csv', False):
        # Use the input CSV filename as base and append "_smoothed" before the extension
        out_csv = path.with_name(f'{path.stem}_smoothed.csv')
        df_sm.to_csv(out_csv, index=False)
        print('Smoothed CSV saved to', out_csv)


if __name__ == '__main__':
    main()
