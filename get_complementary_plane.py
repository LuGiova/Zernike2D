import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import zepyros as zp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from docs import build_cli_complementary_plane


np.seterr(divide='ignore', invalid='ignore')


class ComplementaryPlane:
    def __init__(self, surface_file1, surface_file2, output_path, sample_every=1, use_surface_normals=False, output_name=None):
        self.surface1 = pd.read_csv(surface_file1).reset_index(drop=True)
        self.surface2 = pd.read_csv(surface_file2).reset_index(drop=True)
        self.file_name1 = Path(surface_file1).stem
        self.file_name2 = Path(surface_file2).stem
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.sample_every = sample_every
        self.use_surface_normals = use_surface_normals
        self.output_name = output_name

    @staticmethod
    def get_all_invariants(surface, indices, verso):
        """
        Compute Zernike invariants for each point of a surface.
        """
        coeff_array = np.zeros((len(indices), 121))
        sampled_surface = surface.iloc[indices].reset_index(drop=True)
        z_obj = None

        print(f'Computing Zernike descriptors for {len(sampled_surface)} sampled points')

        for i, ndx in enumerate(tqdm.tqdm(indices, desc='Zernike')):
            coeff, _, _, z_obj = zp.get_zernike(
                surface[['x', 'y', 'z', 'nx', 'ny', 'nz']], 6.0, ndx, 20, int(verso), zernike_obj=z_obj
            )
            coeff_array[i, :] = coeff

        print('Finished Zernike descriptors')

        return sampled_surface, coeff_array

    @staticmethod
    def get_invariants_with_axes(surface, indices, axes_vectors, verso):
        """
        Compute Zernike invariants for given indices using provided axis vectors
        instead of the surface normals.
        """
        coeff_array = np.zeros((len(indices), 121))
        sampled_surface = surface.iloc[indices].reset_index(drop=True)
        z_obj = None

        print(f'Computing Zernike descriptors for {len(indices)} sampled points with custom axes')

        base_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i, ndx in enumerate(tqdm.tqdm(indices, desc='Zernike')):
            # prepare a temporary surface frame where the center normal is replaced by the axis
            surf_mod = surface[base_cols].copy()
            ax = axes_vectors[i]
            # handle degenerate axis
            if np.allclose(ax, 0):
                # keep original normal
                pass
            else:
                axn = ax / np.linalg.norm(ax)
                surf_mod.at[ndx, 'nx'] = axn[0]
                surf_mod.at[ndx, 'ny'] = axn[1]
                surf_mod.at[ndx, 'nz'] = axn[2]

            coeff, _, _, z_obj = zp.get_zernike(
                surf_mod, 6.0, ndx, 20, int(verso), zernike_obj=z_obj
            )
            coeff_array[i, :] = coeff

        print('Finished Zernike descriptors with custom axes')
        return sampled_surface, coeff_array

    @staticmethod
    def fit_plane(midpoints):
        """
        Fit a plane to the midpoint cloud using PCA.
        Returns centroid, basis vectors and projected 3D points.
        """
        if len(midpoints) < 3:
            raise ValueError('At least 3 midpoint points are required to fit a plane')

        print(f'Fitting plane to {len(midpoints)} midpoint points')
        pca = PCA(n_components=3)
        pca.fit(midpoints)
        centroid = pca.mean_
        basis = pca.components_

        centered = midpoints - centroid
        plane_u = centered @ basis[0]
        plane_v = centered @ basis[1]
        projected = centroid + np.outer(plane_u, basis[0]) + np.outer(plane_v, basis[1])

        return centroid, basis, projected, plane_u, plane_v

    @staticmethod
    def plot_plane_subplots(df_plane, phys_col, zernike_col, output_file, cmap='viridis'):
        """
        Create a single image with two subplots (physical and zernike), using the same colormap.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # physical distance subplot
        sc0 = axes[0].scatter(
            df_plane['plane_u'], df_plane['plane_v'],
            c=df_plane[phys_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
        )
        axes[0].set_title('Complementary plane colored by physical distance')
        axes[0].set_xlabel('Plane coordinate u')
        axes[0].set_ylabel('Plane coordinate v')
        # set equal aspect only if ranges are non-degenerate
        umin, umax = df_plane['plane_u'].min(), df_plane['plane_u'].max()
        vmin, vmax = df_plane['plane_v'].min(), df_plane['plane_v'].max()
        if not np.isclose(umax - umin, 0.0) and not np.isclose(vmax - vmin, 0.0):
            axes[0].set_aspect('equal', adjustable='box')
        axes[0].grid(True, alpha=0.25)

        # zernike distance subplot
        sc1 = axes[1].scatter(
            df_plane['plane_u'], df_plane['plane_v'],
            c=df_plane[zernike_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
        )
        axes[1].set_title('Complementary plane colored by Zernike distance')
        axes[1].set_xlabel('Plane coordinate u')
        axes[1].set_ylabel('Plane coordinate v')
        if not np.isclose(umax - umin, 0.0) and not np.isclose(vmax - vmin, 0.0):
            axes[1].set_aspect('equal', adjustable='box')
        axes[1].grid(True, alpha=0.25)

        # colorbars (independent scales but same cmap)
        cbar0 = fig.colorbar(sc0, ax=axes[0])
        cbar0.set_label('Physical distance (Å)')
        cbar1 = fig.colorbar(sc1, ax=axes[1])
        cbar1.set_label('Zernike distance')

        fig.suptitle('Complementary plane: physical vs Zernike')
        fig.savefig(output_file, dpi=180, bbox_inches='tight')
        plt.close(fig)

    def compute(self, plot=False):
        print(f'Loading surfaces: {self.file_name1} and {self.file_name2}')
        if self.sample_every < 1:
            raise ValueError('sample_every must be >= 1')


        sampled_indices1 = list(range(0, len(self.surface1.index), self.sample_every))
        print(f'Sampling every {self.sample_every} point(s) on the first surface')

        # build sampled_surface1 and compute nearest neighbors against the full second surface
        sampled_surface1 = self.surface1.iloc[sampled_indices1].reset_index(drop=True)
        coords1_sample = sampled_surface1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords1_full = self.surface1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords2_full = self.surface2[['x', 'y', 'z']].to_numpy(dtype=float)

        if len(coords1_full) == 0 or len(coords2_full) == 0:
            raise ValueError('Both input surfaces must contain at least one point')

        print(f'Computing nearest neighbors: sampled {self.file_name1} -> full {self.file_name2}')
        dist_matrix_1 = cdist(coords1_sample, coords2_full)
        nearest_idx2 = np.argmin(dist_matrix_1, axis=1)

        initial_pairs = list(zip(sampled_indices1, nearest_idx2.tolist()))
        paired_coords2 = coords2_full[nearest_idx2]
        midpoints_1 = (coords1_sample + paired_coords2) / 2.0

        # now sample surface2 and compute nearest neighbors against the full surface1
        sampled_indices2 = list(range(0, len(self.surface2.index), self.sample_every))
        print(f'Sampling every {self.sample_every} point(s) on the second surface')
        coords2_sample = self.surface2.iloc[sampled_indices2][['x','y','z']].to_numpy(dtype=float)
        dist_matrix_2 = cdist(coords2_sample, coords1_full)
        nearest_idx1 = np.argmin(dist_matrix_2, axis=1)

        inverse_pairs = list(zip(nearest_idx1.tolist(), sampled_indices2))

        # find extra pairs present in inverse matching but not in initial matching
        set_initial = set(initial_pairs)
        set_inverse = set(inverse_pairs)
        extra_pairs = list(set_inverse - set_initial)
        if len(extra_pairs) > 0:
            print(f'Found {len(extra_pairs)} extra pair(s) from inverse matching; adding to analysis')

        # combine pairs (keep original order first, then extras)
        combined_pairs = initial_pairs + extra_pairs
        idx1s = np.array([p[0] for p in combined_pairs], dtype=int)
        idx2s = np.array([p[1] for p in combined_pairs], dtype=int)

        coords1_pts = coords1_full[idx1s]
        coords2_pts = coords2_full[idx2s]
        physical_distance = np.linalg.norm(coords1_pts - coords2_pts, axis=1)
        midpoints = (coords1_pts + coords2_pts) / 2.0

        # fit plane on the combined midpoints
        centroid, basis, projected, plane_u, plane_v = self.fit_plane(midpoints)

        # build axes for each pair (either surface normals or connection vectors)
        if self.use_surface_normals:
            axes1 = self.surface1.loc[idx1s, ['nx', 'ny', 'nz']].to_numpy(dtype=float)
            axes2 = self.surface2.loc[idx2s, ['nx', 'ny', 'nz']].to_numpy(dtype=float)
        else:
            axes1 = coords2_pts - coords1_pts
            axes2 = coords1_pts - coords2_pts
        # normalize axes
        if axes1.size > 0:
            norms1 = np.linalg.norm(axes1, axis=1)
            norms1[norms1 == 0] = 1.0
            axes1 = axes1 / norms1[:, None]
        if axes2.size > 0:
            norms2 = np.linalg.norm(axes2, axis=1)
            norms2[norms2 == 0] = 1.0
            axes2 = axes2 / norms2[:, None]

        # compute Zernike descriptors for all combined pairs (surface1 and surface2)
        _, coeff1 = self.get_invariants_with_axes(self.surface1, idx1s.tolist(), axes1, verso=1)
        _, coeff2 = self.get_invariants_with_axes(self.surface2, idx2s.tolist(), axes2, verso=-1)
        zernike_distance = np.linalg.norm(coeff1 - coeff2, axis=1)

        

        print('Building output table')

        # include original indices and coordinates so downstream smoothing can use 3D points
        df_out = pd.DataFrame({
            'res1': self.surface1.iloc[idx1s]['res'].to_numpy(),
            'res2': self.surface2.iloc[idx2s]['res'].to_numpy(),
            'idx1': idx1s,
            'idx2': idx2s,
            'x1': coords1_pts[:, 0],
            'y1': coords1_pts[:, 1],
            'z1': coords1_pts[:, 2],
            'x2': coords2_pts[:, 0],
            'y2': coords2_pts[:, 1],
            'z2': coords2_pts[:, 2],
            'mid_x': midpoints[:, 0],
            'mid_y': midpoints[:, 1],
            'mid_z': midpoints[:, 2],
            'plane_x': projected[:, 0],
            'plane_y': projected[:, 1],
            'plane_z': projected[:, 2],
            'plane_u': plane_u,
            'plane_v': plane_v,
            'physical_distance': physical_distance,
            'zernike_distance': zernike_distance,
        })

        if self.output_name:
            output_csv = self.output_path / f'{self.output_name}.csv'
        else:
            output_csv = self.output_path / f'{self.file_name1}_{self.file_name2}_complementary_plane.csv'
        df_out.to_csv(output_csv, index=False)
        print(f'Output saved to {output_csv}')
        print(f'Plane centroid: {centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}')
        print(f'Plane normal: {basis[2][0]:.6f}, {basis[2][1]:.6f}, {basis[2][2]:.6f}')

        if plot:
            print('Generating plane plots')
            if self.output_name:
                combined_plot = self.output_path / f'{self.output_name}.png'
            else:
                combined_plot = self.output_path / f'{self.file_name1}_{self.file_name2}_plane_comparison.png'
            self.plot_plane_subplots(
                df_out,
                'physical_distance',
                'zernike_distance',
                combined_plot,
                cmap='viridis'
            )
            print(f'Combined subplot saved to {combined_plot}')

def main():
    args = build_cli_complementary_plane()
    calculator = ComplementaryPlane(
        args.surface1,
        args.surface2,
        args.output,
        sample_every=args.sample_every,
        use_surface_normals=getattr(args, 'use_surface_normals', False)
    )
    calculator.output_name = getattr(args, 'output_name', None)
    calculator.compute(plot=args.plot)


if __name__ == '__main__':
    main()