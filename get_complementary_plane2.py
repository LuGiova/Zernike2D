import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import zepyros as zp
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


from docs import build_cli_complementary_plane2
from get_binding_site import BindingSite


np.seterr(divide='ignore', invalid='ignore')


class ComplementaryPlane:
    def __init__(self, surface_file1, surface_file2, output_path, threshold=5.0, points=100, output_name=None, verbose=False):
        self.surface1 = pd.read_csv(surface_file1).reset_index(drop=True)
        self.surface2 = pd.read_csv(surface_file2).reset_index(drop=True)
        self.file_name1 = Path(surface_file1).stem
        self.file_name2 = Path(surface_file2).stem
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.points = points
        self.output_name = output_name
        self.verbose = verbose

    @staticmethod
    def get_binding_sites(surface1, surface2, threshold):
        coords1 = surface1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords2 = surface2[['x', 'y', 'z']].to_numpy(dtype=float)

        mask1, _ = BindingSite.get_binding_site_mask(coords1, coords2, threshold)
        mask2, _ = BindingSite.get_binding_site_mask(coords2, coords1, threshold)

        return surface1[mask1 == 1].reset_index(drop=True), surface2[mask2 == 1].reset_index(drop=True)

    @staticmethod
    def fit_plane(midpoints):
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
    def plane_points_to_3d(plane_uv, centroid, basis):
        return centroid + np.outer(plane_uv[:, 0], basis[0]) + np.outer(plane_uv[:, 1], basis[1])

    @staticmethod
    def project_surface_to_plane(surface, centroid, basis):
        coords = surface[['x', 'y', 'z']].to_numpy(dtype=float)
        centered = coords - centroid
        plane_u = centered @ basis[0]
        plane_v = centered @ basis[1]
        plane_coords = np.column_stack((plane_u, plane_v))
        projected = centroid + np.outer(plane_u, basis[0]) + np.outer(plane_v, basis[1])
        return coords, plane_coords, projected

    @staticmethod
    def project_point_to_plane(point, centroid, basis):
        point = np.asarray(point, dtype=float)
        centered = point - centroid
        plane_u = float(centered @ basis[0])
        plane_v = float(centered @ basis[1])
        projected = centroid + plane_u * basis[0] + plane_v * basis[1]
        return np.array([plane_u, plane_v], dtype=float), projected

    @staticmethod
    def segment_plane_intersection(point1, point2, plane_point, plane_normal):
        point1 = np.asarray(point1, dtype=float)
        point2 = np.asarray(point2, dtype=float)
        plane_point = np.asarray(plane_point, dtype=float)
        plane_normal = np.asarray(plane_normal, dtype=float)

        direction = point2 - point1
        denominator = float(np.dot(plane_normal, direction))
        if np.isclose(denominator, 0.0):
            return (point1 + point2) / 2.0

        t = float(np.dot(plane_normal, plane_point - point1) / denominator)
        return point1 + t * direction

    @staticmethod
    def build_concentric_rings(plane_coords1, plane_coords2, center_uv, n_rings=10, min_outer_points=10):
        center_uv = np.asarray(center_uv, dtype=float)
        radii1 = np.linalg.norm(plane_coords1 - center_uv, axis=1)
        radii2 = np.linalg.norm(plane_coords2 - center_uv, axis=1)

        max_radius = float(max(np.max(radii1), np.max(radii2)))
        if np.isclose(max_radius, 0.0):
            raise ValueError('Projected binding sites are degenerate in the complementary plane')

        radius = max_radius
        for _ in range(200):
            ring_width = radius / float(n_rings)
            if np.isclose(ring_width, 0.0):
                break

            ring_ids1 = np.full(len(radii1), -1, dtype=int)
            ring_ids2 = np.full(len(radii2), -1, dtype=int)

            inside1 = radii1 <= radius + 1e-12
            inside2 = radii2 <= radius + 1e-12
            ring_ids1[inside1] = np.minimum(np.floor(radii1[inside1] / ring_width).astype(int), n_rings - 1)
            ring_ids2[inside2] = np.minimum(np.floor(radii2[inside2] / ring_width).astype(int), n_rings - 1)

            outer_count1 = int(np.count_nonzero(ring_ids1 == (n_rings - 1)))
            outer_count2 = int(np.count_nonzero(ring_ids2 == (n_rings - 1)))
            if outer_count1 >= min_outer_points and outer_count2 >= min_outer_points:
                return radius, ring_width, radii1, radii2, ring_ids1, ring_ids2

            radius *= 0.95

        raise ValueError('Unable to find a circle radius where the outer ring contains at least 10 points for both binding sites')

    @staticmethod
    def select_ring_pairs(plane_coords1, plane_coords2, coords1, coords2, center_uv, plane_point, plane_normal, basis, radius, ring_ids1, ring_ids2, points_per_ring, n_rings=10):
        center_uv = np.asarray(center_uv, dtype=float)
        relative1 = plane_coords1 - center_uv
        relative2 = plane_coords2 - center_uv
        angle1 = np.arctan2(relative1[:, 1], relative1[:, 0])
        angle2 = np.arctan2(relative2[:, 1], relative2[:, 0])

        records = []
        for ring_id in range(n_rings):
            indices1 = np.flatnonzero(ring_ids1 == ring_id)
            indices2 = np.flatnonzero(ring_ids2 == ring_id)
            if len(indices1) == 0 or len(indices2) == 0:
                continue

            order1 = indices1[np.argsort(angle1[indices1])]
            target_count = min(points_per_ring, len(order1), len(indices2))
            if target_count < 1:
                continue

            if len(order1) > target_count:
                chosen1 = order1[np.linspace(0, len(order1) - 1, target_count, dtype=int)]
            else:
                chosen1 = order1

            distance_matrix = np.linalg.norm(
                plane_coords1[chosen1][:, None, :] - plane_coords2[indices2][None, :, :],
                axis=2,
            )
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
            chosen1 = chosen1[row_ind]
            chosen2 = indices2[col_ind]

            for idx1, idx2 in zip(chosen1, chosen2):
                proj1 = plane_coords1[idx1]
                proj2 = plane_coords2[idx2]
                intersection_3d = ComplementaryPlane.segment_plane_intersection(coords1[idx1], coords2[idx2], plane_point, plane_normal)
                intersection_uv, intersection_proj = ComplementaryPlane.project_point_to_plane(intersection_3d, plane_point, basis)
                relative_intersection = intersection_uv - center_uv
                records.append({
                    'idx1': int(idx1),
                    'idx2': int(idx2),
                    'ring_id': int(ring_id + 1),
                    'ring_fraction': float((ring_id + 1) / n_rings),
                    'circle_radius': float(radius),
                    'ring_width': float(radius / float(n_rings)),
                    'ring_inner_radius': float(ring_id * (radius / float(n_rings))),
                    'ring_outer_radius': float((ring_id + 1) * (radius / float(n_rings))),
                    'plane_u1': float(proj1[0]),
                    'plane_v1': float(proj1[1]),
                    'plane_u2': float(proj2[0]),
                    'plane_v2': float(proj2[1]),
                    'plane_u': float(intersection_uv[0]),
                    'plane_v': float(intersection_uv[1]),
                    'rep_x': float(intersection_proj[0]),
                    'rep_y': float(intersection_proj[1]),
                    'rep_z': float(intersection_proj[2]),
                    'theta': float(np.arctan2(relative_intersection[1], relative_intersection[0])),
                    'radial_distance': float(np.linalg.norm(relative_intersection)),
                    'ring_radius1': float(np.linalg.norm(proj1 - center_uv)),
                    'ring_radius2': float(np.linalg.norm(proj2 - center_uv)),
                })

        return pd.DataFrame.from_records(records)

    @staticmethod
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

    @staticmethod
    def plot_plane_subplots(df_plane, phys_col, zernike_col, output_file, cmap='viridis', circle_center=None, circle_radius=None):
        """
        Create a single image with two subplots (physical and zernike), using the same
        colormap and scatter-based plotting like `get_complementary_plane.py`.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': 'polar'})

        # drop rows with NaN in the plotted columns
        dfp = df_plane.dropna(subset=[phys_col, zernike_col]).reset_index(drop=True)

        if circle_center is None:
            circle_center = np.array([dfp['center_u'].iloc[0], dfp['center_v'].iloc[0]], dtype=float)
        else:
            circle_center = np.asarray(circle_center, dtype=float)

        if circle_radius is None:
            circle_radius = float(dfp['circle_radius'].iloc[0])

        theta = dfp['theta'].to_numpy(dtype=float)
        radial = dfp['radial_distance'].to_numpy(dtype=float)
        ring_ticks = np.linspace(circle_radius / 10.0, circle_radius, 10)
        theta_ticks = np.deg2rad(np.arange(0, 360, 30))
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

        # physical distance subplot
        sc0 = axes[0].scatter(
            theta,
            radial,
            c=dfp[phys_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
        )
        axes[0].set_title('Complementary plane colored by physical distance')

        # zernike distance subplot
        sc1 = axes[1].scatter(
            theta,
            radial,
            c=dfp[zernike_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
        )
        axes[1].set_title('Complementary plane colored by Zernike distance')

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

        centroid, basis, _, plane_u_mid, plane_v_mid = self.fit_plane(midpoints)
        plane_normal = basis[2]
        center_uv, center_proj = self.project_point_to_plane(binding_site_centroid, centroid, basis)
        _, plane_coords1, _ = self.project_surface_to_plane(bs1, centroid, basis)
        _, plane_coords2, _ = self.project_surface_to_plane(bs2, centroid, basis)

        if self.verbose:
            print('Projecting the binding-site centroid on the plane and building concentric rings')

        circle_radius, ring_width, radii1, radii2, ring_ids1, ring_ids2 = self.build_concentric_rings(
            plane_coords1,
            plane_coords2,
            center_uv,
            n_rings=10,
            min_outer_points=10,
        )

        if self.verbose:
            print(f'Circle center projected at u={center_uv[0]:.6f}, v={center_uv[1]:.6f}')
            print(f'Final circle radius={circle_radius:.6f}; ring width={ring_width:.6f}')

        sampled_pairs = self.select_ring_pairs(
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
        plane_uv_sel = sampled_pairs[['plane_u', 'plane_v']].to_numpy(dtype=float)
        plane_xyz_sel = sampled_pairs[['rep_x', 'rep_y', 'rep_z']].to_numpy(dtype=float)
        physical_distance = np.linalg.norm(coords_bs1[idx1] - coords_bs2[idx2], axis=1)

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

        print('Building output table')
        df_out = sampled_pairs.copy()
        df_out['res1'] = bs1.iloc[idx1]['res'].to_numpy()
        df_out['res2'] = bs2.iloc[idx2]['res'].to_numpy()
        df_out['x1'] = coords_bs1[idx1][:, 0]
        df_out['y1'] = coords_bs1[idx1][:, 1]
        df_out['z1'] = coords_bs1[idx1][:, 2]
        df_out['x2'] = coords_bs2[idx2][:, 0]
        df_out['y2'] = coords_bs2[idx2][:, 1]
        df_out['z2'] = coords_bs2[idx2][:, 2]
        df_out['mid_x'] = df_out['rep_x']
        df_out['mid_y'] = df_out['rep_y']
        df_out['mid_z'] = df_out['rep_z']
        df_out['center_u'] = np.full(len(df_out), center_uv[0])
        df_out['center_v'] = np.full(len(df_out), center_uv[1])
        df_out['center_x'] = np.full(len(df_out), center_proj[0])
        df_out['center_y'] = np.full(len(df_out), center_proj[1])
        df_out['center_z'] = np.full(len(df_out), center_proj[2])
        df_out['plane_x'] = plane_xyz_sel[:, 0]
        df_out['plane_y'] = plane_xyz_sel[:, 1]
        df_out['plane_z'] = plane_xyz_sel[:, 2]
        df_out['physical_distance'] = physical_distance
        df_out['zernike_distance'] = zernike_distance

        if self.output_name:
            output_csv = self.output_path / f'{self.output_name}.csv'
        else:
            output_csv = self.output_path / f'{self.file_name1}_{self.file_name2}_complementary_plane.csv'
        df_out.to_csv(output_csv, index=False)
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
            self.plot_plane_subplots(
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
        verbose=getattr(args, 'verbose', False),
    )
    calculator.compute(plot=args.plot)


if __name__ == '__main__':
    main()