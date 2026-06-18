import numpy as np
import pandas as pd
from pathlib import Path

import tqdm
import zepyros as zp
from scipy.spatial import cKDTree

from binding_site_utils import get_binding_site_mask
from complementary_plane_cli import build_cli_complementary_plane2
from plane_geometry import (
    build_concentric_rings,
    fit_plane,
    normal_stats,
    project_point_to_plane,
    project_surface_to_plane,
    select_ring_pairs,
    select_ring_pairs_angular_cells,
    select_ring_pairs_kmeans,
    weighted_stats,
)
from plane_plotting import plot_plane_subplots
from surface_processing import (
    calculate_flatness,
    load_surface_input,
)


np.seterr(divide='ignore', invalid='ignore')


class ComplementaryPlane:
    def __init__(self, surface_file1, surface_file2, output_path, threshold=5.0, points=100, output_name=None, sampling_strategy='default', verbose=False):
        self.surface_file1 = Path(surface_file1)
        self.surface_file2 = Path(surface_file2)
        self.file_name1 = self.surface_file1.stem
        self.file_name2 = self.surface_file2.stem
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.points = points
        self.output_name = output_name
        self.sampling_strategy = sampling_strategy
        self.verbose = verbose
        self.surface1, self.surface1_info = load_surface_input(self.surface_file1)
        self.surface2, self.surface2_info = load_surface_input(self.surface_file2)

    @staticmethod
    def get_binding_sites(surface1, surface2, threshold):
        coords1 = surface1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords2 = surface2[['x', 'y', 'z']].to_numpy(dtype=float)

        mask1, _ = get_binding_site_mask(coords1, coords2, threshold)
        mask2, _ = get_binding_site_mask(coords2, coords1, threshold)

        return surface1[mask1 == 1].reset_index(drop=True), surface2[mask2 == 1].reset_index(drop=True)

    @staticmethod
    # Adapted from get_zernike2d_invariants.py in the original Zernike2D workflow
    # by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
    # Giorgio Gosti, and Giancarlo Ruocco.
    # Source repository: https://github.com/matmi8/Zernike2D.git
    def get_invariants_with_axes(surface, indices, axes_vectors, verso):
        coeff_array = np.zeros((len(indices), 121))
        sampled_surface = surface.iloc[indices].reset_index(drop=True)
        z_obj = None

        print(f'Computing Zernike descriptors for {len(indices)} sampled points with custom axes')

        base_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i, ndx in enumerate(tqdm.tqdm(indices, desc='Zernike')):
            surf_mod = surface[base_cols].copy()
            ax = axes_vectors[i]
            if not np.allclose(ax, 0):
                axn = ax / np.linalg.norm(ax)
                surf_mod.at[ndx, 'nx'] = axn[0]
                surf_mod.at[ndx, 'ny'] = axn[1]
                surf_mod.at[ndx, 'nz'] = axn[2]

            coeff, _, _, z_obj = zp.get_zernike(surf_mod, 6.0, ndx, 20, int(verso), zernike_obj=z_obj)
            coeff_array[i, :] = coeff

        print('Finished Zernike descriptors with custom axes')
        return sampled_surface, coeff_array

    def compute(self, plot=False, save_csv=False):
        print(f'Loading surfaces: {self.file_name1} and {self.file_name2}')
        print(f'Extracting binding sites with threshold {self.threshold} Å')

        bs1, bs2 = self.get_binding_sites(self.surface1, self.surface2, self.threshold)
        if len(bs1) < 1 or len(bs2) < 1:
            raise ValueError('Binding site extraction produced an empty surface')

        # The full surfaces are no longer needed after binding-site extraction.
        self.surface1 = None
        self.surface2 = None

        coords_bs1 = bs1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords_bs2 = bs2[['x', 'y', 'z']].to_numpy(dtype=float)
        combined_coords = np.vstack((coords_bs1, coords_bs2))
        binding_site_centroid = combined_coords.mean(axis=0)

        print(f'Binding site points: {self.file_name1}={len(bs1)}, {self.file_name2}={len(bs2)}')
        print('Computing nearest neighbors between the two binding sites (both directions)')
        # nearest neighbors from bs1 -> bs2
        tree_bs2 = cKDTree(coords_bs2)
        _, nearest_idx2 = tree_bs2.query(coords_bs1, k=1)
        paired_coords2 = coords_bs2[nearest_idx2]
        midpoints1 = (coords_bs1 + paired_coords2) / 2.0

        # nearest neighbors from bs2 -> bs1 (ensure symmetric sampling)
        tree_bs1 = cKDTree(coords_bs1)
        _, nearest_idx1 = tree_bs1.query(coords_bs2, k=1)
        paired_coords1 = coords_bs1[nearest_idx1]
        midpoints2 = (paired_coords1 + coords_bs2) / 2.0

        # combine midpoints from both directions
        midpoints = np.vstack((midpoints1, midpoints2))
        print(f'Collected midpoints: bs1->bs2={len(midpoints1)}, bs2->bs1={len(midpoints2)}, total={len(midpoints)}')

        centroid, basis, _, _, _ = fit_plane(midpoints)
        plane_normal = basis[2]
        center_uv, center_proj = project_point_to_plane(binding_site_centroid, centroid, basis)
        _, plane_coords1, _ = project_surface_to_plane(bs1, centroid, basis)
        _, plane_coords2, _ = project_surface_to_plane(bs2, centroid, basis)

        if self.verbose:
            print('Projecting the binding-site centroid on the plane and building concentric rings')

        circle_radius, ring_width, _, _, ring_ids1, ring_ids2 = build_concentric_rings(
            plane_coords1,
            plane_coords2,
            center_uv,
            n_rings=10,
            min_outer_points=10,
        )
        if self.verbose:
            print(f'Circle center projected at u={center_uv[0]:.6f}, v={center_uv[1]:.6f}')
            print(f'Final circle radius={circle_radius:.6f}; ring width={ring_width:.6f}')

        # Choose sampling strategy
        if self.sampling_strategy == 'angular_cells':
            if self.verbose:
                print(f'Using angular cells sampling strategy with {self.points} target cells per ring')
            sampled_pairs = select_ring_pairs_angular_cells(
                plane_coords1,
                plane_coords2,
                coords_bs1,
                coords_bs2,
                center_uv,
                centroid,
                plane_normal,
                basis,
                circle_radius,
                ring_ids1,
                ring_ids2,
                self.points,
                n_rings=10,
            )
        elif self.sampling_strategy == 'kmeans':
            if self.verbose:
                print(f'Using K-Means sampling strategy with {self.points} clusters per ring')
            sampled_pairs = select_ring_pairs_kmeans(
                plane_coords1,
                plane_coords2,
                coords_bs1,
                coords_bs2,
                center_uv,
                centroid,
                plane_normal,
                basis,
                circle_radius,
                ring_ids1,
                ring_ids2,
                self.points,
                n_rings=10,
            )
        else:  # default
            sampled_pairs = select_ring_pairs(
                plane_coords1,
                plane_coords2,
                coords_bs1,
                coords_bs2,
                center_uv,
                centroid,
                plane_normal,
                basis,
                circle_radius,
                ring_ids1,
                ring_ids2,
                self.points,
                n_rings=10,
            )

        if sampled_pairs.empty:
            raise ValueError('Ring-based sampling produced no matched pairs')

        if self.verbose:
            ring_summary = sampled_pairs.groupby('ring_id').size().to_dict()
            print(f'Requested points per ring: {self.points}; selected pairs per ring: {ring_summary}')

        idx1 = sampled_pairs['idx1'].to_numpy(dtype=int)
        idx2 = sampled_pairs['idx2'].to_numpy(dtype=int)
        physical_distance = np.linalg.norm(coords_bs1[idx1] - coords_bs2[idx2], axis=1)
        del plane_coords1, plane_coords2  # Free memory

        unique_idx1, inverse_idx1 = np.unique(idx1, return_inverse=True)
        unique_idx2, inverse_idx2 = np.unique(idx2, return_inverse=True)

        if self.verbose:
            print(f'Matched points: {len(idx1)}')
            print(f'Unique indices used: surface1={len(unique_idx1)}, surface2={len(unique_idx2)}')

        _, coeff1_unique = self.get_invariants_with_axes(bs1, unique_idx1.tolist(), np.repeat(plane_normal[None, :], len(unique_idx1), axis=0), verso=1)
        _, coeff2_unique = self.get_invariants_with_axes(bs2, unique_idx2.tolist(), np.repeat((-plane_normal)[None, :], len(unique_idx2), axis=0), verso=-1)
        coeff1 = coeff1_unique[inverse_idx1]
        coeff2 = coeff2_unique[inverse_idx2]
        zernike_distance = np.linalg.norm(coeff1 - coeff2, axis=1)
        del coeff1_unique, coeff2_unique, unique_idx1, unique_idx2, inverse_idx1, inverse_idx2  # Free memory

        pc3 = (np.abs(coeff1[:, 2]) + np.abs(coeff2[:, 2])) / 2.0
        normals1 = bs1.iloc[idx1][['nx', 'ny', 'nz']].to_numpy(dtype=float)
        normals2 = bs2.iloc[idx2][['nx', 'ny', 'nz']].to_numpy(dtype=float)
        scalar_prod = np.sum(normals1 * normals2, axis=1)
        del ring_ids1, ring_ids2  # Free memory

        flatness1 = calculate_flatness(coords_bs1)
        flatness2 = calculate_flatness(coords_bs2)
        flatness = float(np.nanmean([flatness1, flatness2]))

        gyration_values = [self.surface1_info.get('gyration_radius', np.nan), self.surface2_info.get('gyration_radius', np.nan)]
        gyration_radius = float(np.nanmean(gyration_values)) if np.any(np.isfinite(gyration_values)) else float('nan')
        if self.surface1_info.get('kind') == 'csv' and self.surface2_info.get('kind') == 'csv':
            gyration_radius_note = 'csv_input'
        elif self.surface1_info.get('kind') == 'pdb' and self.surface2_info.get('kind') == 'pdb':
            gyration_radius_note = 'pdb_mean'
        else:
            gyration_radius_note = 'mixed_input'

        print('Building output table')
        df_out = pd.DataFrame({
            'idx1': idx1,
            'res1': bs1.iloc[idx1]['res'].to_numpy(),
            'x1': coords_bs1[idx1][:, 0],
            'y1': coords_bs1[idx1][:, 1],
            'z1': coords_bs1[idx1][:, 2],
            'idx2': idx2,
            'res2': bs2.iloc[idx2]['res'].to_numpy(),
            'x2': coords_bs2[idx2][:, 0],
            'y2': coords_bs2[idx2][:, 1],
            'z2': coords_bs2[idx2][:, 2],
            'ring_id': sampled_pairs['ring_id'].to_numpy(dtype=int),
            'plane_u1': sampled_pairs['plane_u1'].to_numpy(dtype=float),
            'plane_v1': sampled_pairs['plane_v1'].to_numpy(dtype=float),
            'plane_u2': sampled_pairs['plane_u2'].to_numpy(dtype=float),
            'plane_v2': sampled_pairs['plane_v2'].to_numpy(dtype=float),
            'rep_x': sampled_pairs['rep_x'].to_numpy(dtype=float),
            'rep_y': sampled_pairs['rep_y'].to_numpy(dtype=float),
            'rep_z': sampled_pairs['rep_z'].to_numpy(dtype=float),
            'scalar_prod': scalar_prod,
            'PC3': pc3,
            'physical_distance': physical_distance,
            'zernike_distance': zernike_distance,
        })
        del bs1, bs2, coords_bs1, coords_bs2, sampled_pairs, idx1, idx2, physical_distance, pc3, normals1, normals2, scalar_prod, coeff1, coeff2  # Free memory

        ring_ids = list(range(1, 11))
        summary_rows = []
        
        # Choose which summary types to compute based on sampling strategy
        if self.sampling_strategy == 'default':
            summary_types = ('weighted', 'normal')
        else:
            summary_types = ('normal',)
        
        for summary_type in summary_types:
            row = {}
            for rid in ring_ids:
                ring_sub = df_out[df_out['ring_id'] == rid]
                if ring_sub.empty:
                    row[f'physical_ring{rid}_mean'] = float('nan')
                    row[f'physical_ring{rid}_uncertainty'] = float('nan')
                    row[f'zernike_ring{rid}_mean'] = float('nan')
                    row[f'zernike_ring{rid}_uncertainty'] = float('nan')
                    continue

                ring_plane_coords = ring_sub[['plane_u1', 'plane_v1']].to_numpy(dtype=float)
                if summary_type == 'weighted':
                    phys_mean, phys_unc = weighted_stats(ring_sub['physical_distance'].to_numpy(dtype=float), ring_plane_coords)
                    zern_mean, zern_unc = weighted_stats(ring_sub['zernike_distance'].to_numpy(dtype=float), ring_plane_coords)
                else:
                    phys_mean, phys_unc = normal_stats(ring_sub['physical_distance'].to_numpy(dtype=float))
                    zern_mean, zern_unc = normal_stats(ring_sub['zernike_distance'].to_numpy(dtype=float))

                row[f'physical_ring{rid}_mean'] = phys_mean
                row[f'physical_ring{rid}_uncertainty'] = phys_unc
                row[f'zernike_ring{rid}_mean'] = zern_mean
                row[f'zernike_ring{rid}_uncertainty'] = zern_unc

            all_coords = df_out[['plane_u1', 'plane_v1']].to_numpy(dtype=float)
            if summary_type == 'weighted':
                roughness_mean, roughness_unc = weighted_stats(df_out['PC3'].to_numpy(dtype=float), all_coords)
                scalar_mean, scalar_unc = weighted_stats(df_out['scalar_prod'].to_numpy(dtype=float), all_coords)
            else:
                roughness_mean, roughness_unc = normal_stats(df_out['PC3'].to_numpy(dtype=float))
                scalar_mean, scalar_unc = normal_stats(df_out['scalar_prod'].to_numpy(dtype=float))

            row['gyration_radius'] = gyration_radius
            row['gyration_radius_note'] = gyration_radius_note
            row['flatness'] = flatness
            row['roughness'] = roughness_mean
            row['roughness_uncertainty'] = roughness_unc
            row['scalar_prod'] = scalar_mean
            row['scalar_prod_uncertainty'] = scalar_unc
            row['summary_type'] = summary_type
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_column_order = []
        for rid in ring_ids:
            summary_column_order.extend([
                f'physical_ring{rid}_mean',
                f'physical_ring{rid}_uncertainty',
            ])
        for rid in ring_ids:
            summary_column_order.extend([
                f'zernike_ring{rid}_mean',
                f'zernike_ring{rid}_uncertainty',
            ])
        summary_column_order.extend([
            'gyration_radius',
            'gyration_radius_note',
            'flatness',
            'roughness',
            'roughness_uncertainty',
            'scalar_prod',
            'scalar_prod_uncertainty',
            'summary_type',
        ])
        summary_df = summary_df[summary_column_order]

        if self.output_name:
            stem = self.output_name
        else:
            stem = f'{self.file_name1}_{self.file_name2}'
        summary_path = self.output_path / f'{stem}_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f'Summary saved to {summary_path}')

        if save_csv:
            if self.output_name:
                output_csv = self.output_path / f'{self.output_name}.csv'
            else:
                output_csv = self.output_path / f'{self.file_name1}_{self.file_name2}_complementary_plane.csv'

            meta_lines = [
                f"# center_u,center_v: {center_uv[0]:.6f},{center_uv[1]:.6f}",
                f"# center_x,center_y,center_z: {center_proj[0]:.6f},{center_proj[1]:.6f},{center_proj[2]:.6f}",
                f"# circle_radius: {circle_radius:.6f}",
                f"# ring_width: {ring_width:.6f}",
                f"# n_rings: 10",
            ]

            detailed_df = df_out[['idx1', 'res1', 'x1', 'y1', 'z1', 'idx2', 'res2', 'x2', 'y2', 'z2', 'ring_id', 'plane_u1', 'plane_v1', 'plane_u2', 'plane_v2', 'rep_x', 'rep_y', 'rep_z', 'scalar_prod', 'PC3', 'physical_distance', 'zernike_distance']]

            with open(output_csv, 'w', newline='') as fh:
                for line in meta_lines:
                    fh.write(line + '\n')
                detailed_df.to_csv(fh, index=False)

            print(f'Output saved to {output_csv}')
        print(f'Plane centroid: {centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}')
        print(f'Plane normal: {plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}')
        print(f'Projected binding-site centroid on plane: {center_proj[0]:.6f}, {center_proj[1]:.6f}, {center_proj[2]:.6f}')
        print(f'Circle radius: {circle_radius:.6f} with 10 rings of width {ring_width:.6f}')

        if plot:
            print('Generating plane plots')
            if self.output_name:
                combined_plot = self.output_path / f'{self.output_name}.png'
            else:
                combined_plot = self.output_path / f'{self.file_name1}_{self.file_name2}_plane_comparison.png'
            # Plot only matched points on the plane (no cell rendering, no black unmatched cells)
            plot_plane_subplots(
                df_out,
                'physical_distance',
                'zernike_distance',
                combined_plot,
                cmap='viridis',
                circle_center=center_uv,
                circle_radius=circle_radius,
            )
            print(f'Combined subplot saved to {combined_plot}')
def main():
    args = build_cli_complementary_plane2()
    calculator = ComplementaryPlane(
        args.surface1,
        args.surface2,
        args.output,
        threshold=args.threshold,
        points=args.points,
        output_name=getattr(args, 'output_name', None),
        sampling_strategy=getattr(args, 'sampling_strategy', 'default'),
        verbose=getattr(args, 'verbose', False),
    )
    # run compute; CSV saved only if --csv flag provided; summary saved if requested
    calculator.compute(
        plot=args.plot,
        save_csv=getattr(args, 'csv', False),
    )


if __name__ == '__main__':
    main()