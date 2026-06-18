"""Plotting helpers for complementary-plane diagnostics."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_plane_subplots(df_plane, phys_col, zernike_col, output_file, cmap='viridis', circle_center=None, circle_radius=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': 'polar'})

    dfp = df_plane.dropna(subset=[phys_col, zernike_col]).reset_index(drop=True)

    if 'theta' in dfp.columns and 'radial_distance' in dfp.columns:
        theta = dfp['theta'].to_numpy(dtype=float)
        radial = dfp['radial_distance'].to_numpy(dtype=float)
    else:
        if {'plane_u', 'plane_v'}.issubset(dfp.columns):
            plane_u = dfp['plane_u'].to_numpy(dtype=float)
            plane_v = dfp['plane_v'].to_numpy(dtype=float)
        else:
            plane_u = dfp['plane_u1'].to_numpy(dtype=float)
            plane_v = dfp['plane_v1'].to_numpy(dtype=float)

        if circle_center is None:
            if {'center_u', 'center_v'}.issubset(dfp.columns):
                circle_center = np.array([dfp['center_u'].iloc[0], dfp['center_v'].iloc[0]], dtype=float)
            else:
                circle_center = np.array([0.0, 0.0], dtype=float)
        else:
            circle_center = np.asarray(circle_center, dtype=float)

        theta = np.arctan2(plane_v - circle_center[1], plane_u - circle_center[0])
        radial = np.hypot(plane_u - circle_center[0], plane_v - circle_center[1])

    if circle_center is None:
        circle_center = np.array([dfp['center_u'].iloc[0], dfp['center_v'].iloc[0]], dtype=float)
    else:
        circle_center = np.asarray(circle_center, dtype=float)

    if circle_radius is None:
        circle_radius = float(dfp['circle_radius'].iloc[0])

    ring_ticks = np.linspace(circle_radius / 10.0, circle_radius, 10)
    theta_labels = [f'{deg}°' for deg in range(0, 360, 30)]
    radial_labels = [''] * (len(ring_ticks) - 1) + [f'{circle_radius:.2f}']

    for ax in axes:
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_rlim(0, circle_radius)
        ax.set_rticks(ring_ticks)
        ax.set_yticklabels(radial_labels)
        ax.set_thetagrids(np.arange(0, 360, 30), labels=theta_labels)
        ax.grid(True, alpha=0.35)
        ax.set_rlabel_position(135)

    sc0 = axes[0].scatter(
        theta,
        radial,
        c=dfp[phys_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
    )
    axes[0].set_title('Complementary plane colored by physical distance')

    sc1 = axes[1].scatter(
        theta,
        radial,
        c=dfp[zernike_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
    )
    axes[1].set_title('Complementary plane colored by Zernike distance')

    cbar0 = fig.colorbar(sc0, ax=axes[0])
    cbar0.set_label('Physical distance (Å)')
    cbar1 = fig.colorbar(sc1, ax=axes[1])
    cbar1.set_label('Zernike distance')

    fig.suptitle('Complementary plane: physical vs Zernike')
    fig.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close(fig)