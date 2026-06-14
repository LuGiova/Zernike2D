import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import zepyros as zp
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


from docs import build_cli_complementary_plane2
from get_binding_site import BindingSite


np.seterr(divide='ignore', invalid='ignore')


class ComplementaryPlane:
    def __init__(self, surface_file1, surface_file2, output_path, threshold=5.0, points=100, output_name=None, verbose=False, render_plane_3d=False):
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
        self.render_plane_3d = render_plane_3d

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
    def build_rectangle(plane_u, plane_v):
        umin, umax = float(np.min(plane_u)), float(np.max(plane_u))
        vmin, vmax = float(np.min(plane_v)), float(np.max(plane_v))
        return umin, umax, vmin, vmax

    @staticmethod
    def sample_rectangle_points(umin, umax, vmin, vmax, n_points):
        if n_points < 1:
            raise ValueError('points must be >= 1')

        n_u = max(1, int(np.floor(np.sqrt(n_points))))
        n_v = max(1, int(np.ceil(n_points / n_u)))

        u_centers = np.linspace(umin, umax, n_u + 1)
        v_centers = np.linspace(vmin, vmax, n_v + 1)
        u_centers = (u_centers[:-1] + u_centers[1:]) / 2.0
        v_centers = (v_centers[:-1] + v_centers[1:]) / 2.0

        grid = np.array(np.meshgrid(u_centers, v_centers, indexing='xy')).reshape(2, -1).T
        if len(grid) == n_points:
            return grid

        indices = np.linspace(0, len(grid) - 1, n_points, dtype=int)
        return grid[indices]

    @staticmethod
    def build_rectangle_grid(umin, umax, vmin, vmax, n_points):
        if n_points < 1:
            raise ValueError('points must be >= 1')
        # choose integer grid (n_u, n_v) such that n_u * n_v >= n_points
        # and cells are as square as possible in (u, v) coordinates
        width = float(umax - umin)
        height = float(vmax - vmin)
        if width == 0 or height == 0:
            # degenerate: fallback to 1 x n_points
            n_u, n_v = 1, int(np.ceil(n_points))
        else:
            best_pair = (1, int(np.ceil(n_points)))
            best_cell_log_ratio = None
            best_p = None
            max_u = int(np.ceil(n_points))
            for n_u in range(1, max_u + 1):
                n_v = int(np.ceil(n_points / n_u))
                p = n_u * n_v

                cell_u = width / n_u
                cell_v = height / n_v
                # log-distance from 1.0 gives symmetric penalty for x and 1/x
                cell_log_ratio = abs(np.log(cell_u / cell_v))

                if (
                    best_cell_log_ratio is None
                    or cell_log_ratio < best_cell_log_ratio
                    or (np.isclose(cell_log_ratio, best_cell_log_ratio) and p < best_p)
                ):
                    best_pair = (n_u, n_v)
                    best_cell_log_ratio = cell_log_ratio
                    best_p = p

            n_u, n_v = best_pair

        u_edges = np.linspace(umin, umax, n_u + 1)
        v_edges = np.linspace(vmin, vmax, n_v + 1)
        u_centers = (u_edges[:-1] + u_edges[1:]) / 2.0
        v_centers = (v_edges[:-1] + v_edges[1:]) / 2.0
        grid = np.array(np.meshgrid(u_centers, v_centers, indexing='xy')).reshape(2, -1).T

        return grid, u_edges, v_edges, len(grid), n_u, n_v

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
    def assign_points_to_cells(plane_coords, u_edges, v_edges):
        u_idx = np.digitize(plane_coords[:, 0], u_edges, right=False) - 1
        v_idx = np.digitize(plane_coords[:, 1], v_edges, right=False) - 1

        u_idx = np.clip(u_idx, 0, len(u_edges) - 2)
        v_idx = np.clip(v_idx, 0, len(v_edges) - 2)

        cell_map = {}
        for idx, cell in enumerate(zip(u_idx, v_idx)):
            cell_map.setdefault(cell, []).append(idx)
        return cell_map

    @staticmethod
    def select_cell_matches(rectangle_uv, plane_coords1, plane_coords2, coords1, coords2, u_edges, v_edges, require_both=False, return_rect_idx=False):
        cell_map1 = ComplementaryPlane.assign_points_to_cells(plane_coords1, u_edges, v_edges)
        cell_map2 = ComplementaryPlane.assign_points_to_cells(plane_coords2, u_edges, v_edges)

        chosen_idx1 = []
        chosen_idx2 = []
        both_count = 0
        fallback_count = 0
        rect_indices = []

        for ridx, cell_uv in enumerate(rectangle_uv):
            u_idx = np.searchsorted(u_edges, cell_uv[0], side='right') - 1
            v_idx = np.searchsorted(v_edges, cell_uv[1], side='right') - 1
            u_idx = int(np.clip(u_idx, 0, len(u_edges) - 2))
            v_idx = int(np.clip(v_idx, 0, len(v_edges) - 2))
            cell = (u_idx, v_idx)

            candidates1 = cell_map1.get(cell, [])
            candidates2 = cell_map2.get(cell, [])

            if candidates1 and candidates2:
                pair_coords1 = coords1[candidates1]
                pair_coords2 = coords2[candidates2]
                pair_dist = np.linalg.norm(
                    pair_coords1[:, None, :] - pair_coords2[None, :, :],
                    axis=2
                )
                best = np.unravel_index(np.argmin(pair_dist), pair_dist.shape)
                chosen_idx1.append(candidates1[best[0]])
                chosen_idx2.append(candidates2[best[1]])
                rect_indices.append(ridx)
                both_count += 1
                continue

            if require_both:
                # skip this cell
                continue

            # fallback: if one cell side is empty, pick the nearest projected point to the cell center on each surface
            if not candidates1:
                candidates1 = [int(np.argmin(np.linalg.norm(plane_coords1 - cell_uv, axis=1)))]
            if not candidates2:
                candidates2 = [int(np.argmin(np.linalg.norm(plane_coords2 - cell_uv, axis=1)))]

            fallback_count += 1
            chosen_idx1.append(candidates1[0])
            chosen_idx2.append(candidates2[0])
            rect_indices.append(ridx)

        out1 = np.asarray(chosen_idx1, dtype=int)
        out2 = np.asarray(chosen_idx2, dtype=int)
        if return_rect_idx:
            return out1, out2, np.asarray(rect_indices, dtype=int), both_count, fallback_count
        return out1, out2, both_count, fallback_count

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
    def plot_plane_subplots(df_plane, phys_col, zernike_col, output_file, cmap='viridis', u_edges=None, v_edges=None, n_u=None, n_v=None, phys_grid=None, zernike_grid=None):
        """
        Create a single image with two subplots (physical and zernike), using the same
        colormap and scatter-based plotting like `get_complementary_plane.py`.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # drop rows with NaN in the plotted columns
        dfp = df_plane.dropna(subset=[phys_col, zernike_col]).reset_index(drop=True)

        # physical distance subplot
        sc0 = axes[0].scatter(
            dfp['plane_u'], dfp['plane_v'],
            c=dfp[phys_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
        )
        axes[0].set_title('Complementary plane colored by physical distance')
        axes[0].set_xlabel('Plane coordinate u')
        axes[0].set_ylabel('Plane coordinate v')
        # set equal aspect only if ranges are non-degenerate
        umin, umax = dfp['plane_u'].min(), dfp['plane_u'].max()
        vmin, vmax = dfp['plane_v'].min(), dfp['plane_v'].max()
        if not np.isclose(umax - umin, 0.0) and not np.isclose(vmax - vmin, 0.0):
            axes[0].set_aspect('equal', adjustable='box')
        axes[0].grid(True, alpha=0.25)

        # zernike distance subplot
        sc1 = axes[1].scatter(
            dfp['plane_u'], dfp['plane_v'],
            c=dfp[zernike_col], cmap=cmap, s=18, alpha=0.9, edgecolors='none'
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
        print(f'Extracting binding sites with threshold {self.threshold} Å')

        bs1, bs2 = self.get_binding_sites(self.surface1, self.surface2, self.threshold)
        if len(bs1) < 1 or len(bs2) < 1:
            raise ValueError('Binding site extraction produced an empty surface')

        coords_bs1 = bs1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords_bs2 = bs2[['x', 'y', 'z']].to_numpy(dtype=float)

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
        # if requested, render only the plane and binding-site points and exit
        if self.render_plane_3d:
            if self.output_name:
                out_png = self.output_path / f'{self.output_name}_3d.png'
            else:
                out_png = self.output_path / f'{self.file_name1}_{self.file_name2}_3d.png'
            # binding site coords (use bs1 and bs2 computed earlier)
            bs1_coords = bs1[['x', 'y', 'z']].to_numpy(dtype=float)
            bs2_coords = bs2[['x', 'y', 'z']].to_numpy(dtype=float)
            umin, umax, vmin, vmax = float(np.min(plane_u_mid)), float(np.max(plane_u_mid)), float(np.min(plane_v_mid)), float(np.max(plane_v_mid))
            uu = np.linspace(umin, umax, 40)
            vv = np.linspace(vmin, vmax, 40)
            UU, VV = np.meshgrid(uu, vv)
            uv_grid = np.column_stack((UU.ravel(), VV.ravel()))
            XYZ = centroid + np.outer(uv_grid[:,0], basis[0]) + np.outer(uv_grid[:,1], basis[1])
            XYZ = XYZ.reshape(UU.shape[0], UU.shape[1], 3)
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(bs1_coords[:,0], bs1_coords[:,1], bs1_coords[:,2], c='C0', s=6, label=self.file_name1)
            ax.scatter(bs2_coords[:,0], bs2_coords[:,1], bs2_coords[:,2], c='C1', s=6, label=self.file_name2)
            ax.plot_surface(XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2], color='gray', alpha=0.4)
            ax.set_title('Binding sites and fitted plane')
            ax.legend()
            fig.savefig(out_png, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f'3D render saved to {out_png}')
            # also export glTF
            try:
                save_gltf_mesh(out_png.with_suffix('.gltf'), XYZ, bs1_coords, bs2_coords)
                print(f'3D model saved to {out_png.with_suffix('.gltf')}')
            except Exception as e:
                print('Failed to save glTF:', e)
            return
        # (render handled above) continue full computation
        umin, umax, vmin, vmax = self.build_rectangle(plane_u_mid, plane_v_mid)
        # iterate grids until we get at least `points` cells with both-side matches (no fallback)
        if self.verbose:
            print('Projecting the full surfaces on the plane and matching the sampled rectangle points')
        _, plane_coords1, _ = self.project_surface_to_plane(self.surface1, centroid, basis)
        _, plane_coords2, _ = self.project_surface_to_plane(self.surface2, centroid, basis)

        target = self.points
        max_product = max(10 * self.points, 2000)
        found = False
        iter_count = 0
        chosen_rect_indices = None
        chosen_idx1 = None
        chosen_idx2 = None
        while True:
            rectangle_uv, u_edges, v_edges, actual_count, n_u, n_v = self.build_rectangle_grid(umin, umax, vmin, vmax, target)
            rectangle_xyz = self.plane_points_to_3d(rectangle_uv, centroid, basis)

            out = self.select_cell_matches(
                rectangle_uv,
                plane_coords1,
                plane_coords2,
                self.surface1[['x', 'y', 'z']].to_numpy(dtype=float),
                self.surface2[['x', 'y', 'z']].to_numpy(dtype=float),
                u_edges,
                v_edges,
                require_both=True,
                return_rect_idx=True,
            )
            idx1_tmp, idx2_tmp, rect_idx_tmp, both_count, _ = out

            if self.verbose:
                print(f'Iter {iter_count}: target={target}; grid={n_u}x{n_v} product={actual_count}; both_count={both_count}')

            if both_count >= self.points:
                # n is a minimum: keep all matched cells found at this resolution
                keep = np.argsort(rect_idx_tmp)
                chosen_rect_indices = rect_idx_tmp[keep]
                chosen_idx1 = idx1_tmp[keep]
                chosen_idx2 = idx2_tmp[keep]
                found = True
                break

            # increase target and retry
            iter_count += 1
            if actual_count >= max_product or iter_count > 20:
                # cannot find enough matching cells; fall back to best available (may be < points)
                if len(rect_idx_tmp) > 0:
                    keep = np.argsort(rect_idx_tmp)
                    chosen_rect_indices = rect_idx_tmp[keep]
                    chosen_idx1 = idx1_tmp[keep]
                    chosen_idx2 = idx2_tmp[keep]
                else:
                    # nothing found; use original grid with fallback behavior
                    rectangle_uv, u_edges, v_edges, actual_count, n_u, n_v = self.build_rectangle_grid(umin, umax, vmin, vmax, self.points)
                    rectangle_xyz = self.plane_points_to_3d(rectangle_uv, centroid, basis)
                    chosen_idx1, chosen_idx2, both_count2, fallback_count2 = self.select_cell_matches(
                        rectangle_uv,
                        plane_coords1,
                        plane_coords2,
                        self.surface1[['x', 'y', 'z']].to_numpy(dtype=float),
                        self.surface2[['x', 'y', 'z']].to_numpy(dtype=float),
                        u_edges,
                        v_edges,
                        require_both=False,
                        return_rect_idx=False,
                    )
                    chosen_rect_indices = np.arange(len(rectangle_uv))
                break

            target = int(np.ceil(target * 1.3))

        if self.verbose:
            print(f'Requested points: {self.points}; final grid: n_u={n_u}, n_v={n_v}, product={actual_count}; returned_cells={len(chosen_idx1)}')

        coords1 = self.surface1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords2 = self.surface2[['x', 'y', 'z']].to_numpy(dtype=float)

        # idx arrays are those chosen for the matched rectangle cells
        idx1 = np.asarray(chosen_idx1, dtype=int)
        idx2 = np.asarray(chosen_idx2, dtype=int)
        rect_idx = np.asarray(chosen_rect_indices, dtype=int)
        rectangle_uv_sel = rectangle_uv[rect_idx]
        rectangle_xyz_sel = rectangle_xyz[rect_idx]
        physical_distance = np.linalg.norm(coords1[idx1] - coords2[idx2], axis=1)
        midpoints_sampled = (coords1[idx1] + coords2[idx2]) / 2.0

        plane_normal = basis[2]
        axes1 = np.repeat(plane_normal[None, :], len(idx1), axis=0)
        axes2 = np.repeat((-plane_normal)[None, :], len(idx2), axis=0)

        unique_idx1, inverse_idx1 = np.unique(idx1, return_inverse=True)
        unique_idx2, inverse_idx2 = np.unique(idx2, return_inverse=True)

        if self.verbose:
            print(f'Matched points: {len(idx1)}')
            print(f'Unique indices used: surface1={len(unique_idx1)}, surface2={len(unique_idx2)}')

        _, coeff1_unique = self.get_invariants_with_axes(self.surface1, unique_idx1.tolist(), np.repeat(plane_normal[None, :], len(unique_idx1), axis=0), verso=1)
        _, coeff2_unique = self.get_invariants_with_axes(self.surface2, unique_idx2.tolist(), np.repeat((-plane_normal)[None, :], len(unique_idx2), axis=0), verso=-1)
        coeff1 = coeff1_unique[inverse_idx1]
        coeff2 = coeff2_unique[inverse_idx2]
        zernike_distance = np.linalg.norm(coeff1 - coeff2, axis=1)

        print('Building output table')
        # include original indices and coordinates so downstream smoothing can use 3D points
        df_out = pd.DataFrame({
            'res1': self.surface1.iloc[idx1]['res'].to_numpy(),
            'res2': self.surface2.iloc[idx2]['res'].to_numpy(),
            'idx1': idx1,
            'idx2': idx2,
            'x1': coords1[idx1][:, 0],
            'y1': coords1[idx1][:, 1],
            'z1': coords1[idx1][:, 2],
            'x2': coords2[idx2][:, 0],
            'y2': coords2[idx2][:, 1],
            'z2': coords2[idx2][:, 2],
            'mid_x': midpoints_sampled[:, 0],
            'mid_y': midpoints_sampled[:, 1],
            'mid_z': midpoints_sampled[:, 2],
            'plane_x': rectangle_xyz_sel[:, 0],
            'plane_y': rectangle_xyz_sel[:, 1],
            'plane_z': rectangle_xyz_sel[:, 2],
            'plane_u': rectangle_uv_sel[:, 0],
            'plane_v': rectangle_uv_sel[:, 1],
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
        print(f'Plane normal: {plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}')

        if plot:
            print('Generating plane plots')
            if self.output_name:
                combined_plot = self.output_path / f'{self.output_name}.png'
            else:
                combined_plot = self.output_path / f'{self.file_name1}_{self.file_name2}_plane_comparison.png'
            # Plot only matched points on the plane (no cell rendering, no black unmatched cells)
            self.plot_plane_subplots(df_out, 'physical_distance', 'zernike_distance', combined_plot, cmap='viridis')
            print(f'Combined subplot saved to {combined_plot}')


def save_gltf_mesh(out_path, plane_xyz, pts1, pts2):
    import json, base64
    import numpy as _np
    H, W, _ = plane_xyz.shape
    verts = plane_xyz.reshape(-1, 3).astype(_np.float32)
    idx = []
    for i in range(H-1):
        for j in range(W-1):
            a = i*W + j
            b = a + 1
            c = a + W
            d = c + 1
            idx.extend([a, b, c, b, d, c])
    indices = _np.array(idx, dtype=_np.uint32)

    pts1 = _np.asarray(pts1, dtype=_np.float32)
    pts2 = _np.asarray(pts2, dtype=_np.float32)

    bin_parts = [verts.tobytes(), indices.tobytes(), pts1.tobytes(), pts2.tobytes()]
    offsets = []
    cur = 0
    for p in bin_parts:
        offsets.append(cur)
        cur += len(p)
    blob = b''.join(bin_parts)
    b64 = base64.b64encode(blob).decode('ascii')
    uri = 'data:application/octet-stream;base64,' + b64

    def accessor_min_max(arr):
        return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()

    gltf = {
        'asset': {'version': '2.0'},
        'buffers': [{'uri': uri, 'byteLength': len(blob)}],
        'bufferViews': [],
        'accessors': [],
        'meshes': [],
        'nodes': [],
        'scenes': [{'nodes': [0]}],
        'scene': 0,
    }

    gltf['bufferViews'].append({'buffer': 0, 'byteOffset': offsets[0], 'byteLength': len(bin_parts[0])})
    gltf['bufferViews'].append({'buffer': 0, 'byteOffset': offsets[1], 'byteLength': len(bin_parts[1])})
    gltf['bufferViews'].append({'buffer': 0, 'byteOffset': offsets[2], 'byteLength': len(bin_parts[2])})
    gltf['bufferViews'].append({'buffer': 0, 'byteOffset': offsets[3], 'byteLength': len(bin_parts[3])})

    vmin, vmax = accessor_min_max(verts)
    gltf['accessors'].append({'bufferView': 0, 'byteOffset': 0, 'componentType': 5126, 'count': len(verts), 'type': 'VEC3', 'min': vmin, 'max': vmax})
    gltf['accessors'].append({'bufferView': 1, 'byteOffset': 0, 'componentType': 5125, 'count': len(indices), 'type': 'SCALAR'})
    p1min, p1max = accessor_min_max(pts1) if len(pts1) > 0 else ([0,0,0],[0,0,0])
    p2min, p2max = accessor_min_max(pts2) if len(pts2) > 0 else ([0,0,0],[0,0,0])
    gltf['accessors'].append({'bufferView': 2, 'byteOffset': 0, 'componentType': 5126, 'count': len(pts1), 'type': 'VEC3', 'min': p1min, 'max': p1max})
    gltf['accessors'].append({'bufferView': 3, 'byteOffset': 0, 'componentType': 5126, 'count': len(pts2), 'type': 'VEC3', 'min': p2min, 'max': p2max})

    # materials: 0=plane gray, 1=pts1 color (blue), 2=pts2 color (orange)
    gltf['materials'] = [
        # semi-transparent double-sided plane to avoid occluding the proteins
        {'name': 'plane', 'pbrMetallicRoughness': {'baseColorFactor': [0.7, 0.7, 0.7, 0.25], 'metallicFactor': 0.0, 'roughnessFactor': 1.0}, 'alphaMode': 'BLEND', 'doubleSided': True},
        {'name': 'surface1', 'pbrMetallicRoughness': {'baseColorFactor': [0.0, 0.447, 0.741, 1.0], 'metallicFactor': 0.0, 'roughnessFactor': 1.0}, 'emissiveFactor': [0.0, 0.447, 0.741]},
        {'name': 'surface2', 'pbrMetallicRoughness': {'baseColorFactor': [0.85, 0.325, 0.098, 1.0], 'metallicFactor': 0.0, 'roughnessFactor': 1.0}, 'emissiveFactor': [0.85, 0.325, 0.098]},
    ]

    plane_prim = {'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 4, 'material': 0}
    pts1_prim = {'attributes': {'POSITION': 2}, 'mode': 0, 'material': 1} if len(pts1) > 0 else None
    pts2_prim = {'attributes': {'POSITION': 3}, 'mode': 0, 'material': 2} if len(pts2) > 0 else None
    primitives = [plane_prim]
    if pts1_prim: primitives.append(pts1_prim)
    if pts2_prim: primitives.append(pts2_prim)
    gltf['meshes'].append({'primitives': primitives})

    gltf['nodes'].append({'mesh': 0})

    with open(out_path, 'w') as f:
        json.dump(gltf, f)


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
        render_plane_3d=getattr(args, 'render_plane_3d', False),
    )
    calculator.compute(plot=args.plot)


if __name__ == '__main__':
    main()