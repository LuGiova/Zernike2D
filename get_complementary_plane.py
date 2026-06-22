import gc
import os
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import tqdm
try:
    import zepyros as zp
except ModuleNotFoundError:  # lets --help and batch discovery work even before zepyros is installed
    zp = None
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


@contextmanager
def _suppress_worker_output(enabled):
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, 'wb') as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


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

    def _log(self, message):
        if self.verbose:
            print(message)

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
    def get_invariants_with_axes(surface, indices, axes_vectors, verso, verbose=False):
        if zp is None:
            raise ImportError('The zepyros package is required to compute Zernike descriptors.')

        coeff_array = np.zeros((len(indices), 121))
        sampled_surface = surface.iloc[indices].reset_index(drop=True)
        z_obj = None

        if verbose:
            print(f'Computing Zernike descriptors for {len(indices)} sampled points with custom axes')

        base_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        iterator = tqdm.tqdm(indices, desc='Zernike', disable=not verbose)
        for i, ndx in enumerate(iterator):
            surf_mod = surface[base_cols].copy()
            # Ensure numeric types (important when surface is loaded from CSV or other sources)
            for col in base_cols:
                surf_mod[col] = pd.to_numeric(surf_mod[col], errors='coerce')
            
            ax = axes_vectors[i]
            if not np.allclose(ax, 0):
                axn = ax / np.linalg.norm(ax)
                surf_mod.at[ndx, 'nx'] = axn[0]
                surf_mod.at[ndx, 'ny'] = axn[1]
                surf_mod.at[ndx, 'nz'] = axn[2]

            coeff, _, _, z_obj = zp.get_zernike(surf_mod, 6.0, ndx, 20, int(verso), zernike_obj=z_obj)
            coeff_array[i, :] = coeff

        if verbose:
            print('Finished Zernike descriptors with custom axes')
        return sampled_surface, coeff_array

    def compute(self, plot=False, save_csv=False, save_summary=True):
        """Compute complementary-plane metrics.

        Parameters
        ----------
        plot : bool
            Save diagnostic plots.
        save_csv : bool
            Save detailed per-point CSV.
        save_summary : bool
            Save the usual per-complex summary CSV. Batch mode sets this to False and
            appends the returned summary to one global summary file instead.

        Returns
        -------
        pandas.DataFrame
            Summary rows for this complex.
        """
        self._log(f'Loading surfaces: {self.file_name1} and {self.file_name2}')
        self._log(f'Extracting binding sites with threshold {self.threshold} Å')

        bs1, bs2 = self.get_binding_sites(self.surface1, self.surface2, self.threshold)
        if len(bs1) < 1 or len(bs2) < 1:
            raise ValueError('Binding site extraction produced an empty surface')

        # The full surfaces are no longer needed after binding-site extraction.
        self.surface1 = None
        self.surface2 = None
        gc.collect()

        coords_bs1 = bs1[['x', 'y', 'z']].to_numpy(dtype=float)
        coords_bs2 = bs2[['x', 'y', 'z']].to_numpy(dtype=float)
        combined_coords = np.vstack((coords_bs1, coords_bs2))
        binding_site_centroid = combined_coords.mean(axis=0)
        del combined_coords

        self._log(f'Binding site points: {self.file_name1}={len(bs1)}, {self.file_name2}={len(bs2)}')
        self._log('Computing nearest neighbors between the two binding sites (both directions)')
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
        del tree_bs1, tree_bs2, nearest_idx1, nearest_idx2, paired_coords1, paired_coords2

        # combine midpoints from both directions
        midpoints = np.vstack((midpoints1, midpoints2))
        self._log(f'Collected midpoints: bs1->bs2={len(midpoints1)}, bs2->bs1={len(midpoints2)}, total={len(midpoints)}')
        del midpoints1, midpoints2

        centroid, basis, _, _, _ = fit_plane(midpoints)
        del midpoints
        plane_normal = basis[2]
        center_uv, center_proj = project_point_to_plane(binding_site_centroid, centroid, basis)
        _, plane_coords1, _ = project_surface_to_plane(bs1, centroid, basis)
        _, plane_coords2, _ = project_surface_to_plane(bs2, centroid, basis)
        bs1_pc3 = (coords_bs1 - centroid) @ plane_normal
        bs2_pc3 = (coords_bs2 - centroid) @ plane_normal
        bs1_pc3_mean = float(np.mean(bs1_pc3))
        bs2_pc3_mean = float(np.mean(bs2_pc3))
        bs1_pc3_var = float(np.var(bs1_pc3))
        bs2_pc3_var = float(np.var(bs2_pc3))

        self._log('Projecting the binding-site centroid on the plane and building concentric rings')
        circle_radius, ring_width, _, _, ring_ids1, ring_ids2 = build_concentric_rings(
            plane_coords1,
            plane_coords2,
            center_uv,
            n_rings=10,
            min_outer_points=10,
        )
        self._log(f'Circle center projected at u={center_uv[0]:.6f}, v={center_uv[1]:.6f}')
        self._log(f'Final circle radius={circle_radius:.6f}; ring width={ring_width:.6f}')

        # Choose sampling strategy
        if self.sampling_strategy == 'angular_cells':
            self._log(f'Using angular cells sampling strategy with {self.points} target cells per ring')
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
            self._log(f'Using K-Means sampling strategy with {self.points} clusters per ring')
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
        pc3_1 = bs1_pc3[idx1]
        pc3_2 = bs2_pc3[idx2]
        del plane_coords1, plane_coords2  # Free memory

        unique_idx1, inverse_idx1 = np.unique(idx1, return_inverse=True)
        unique_idx2, inverse_idx2 = np.unique(idx2, return_inverse=True)

        if self.verbose:
            print(f'Matched points: {len(idx1)}')
            print(f'Unique indices used: surface1={len(unique_idx1)}, surface2={len(unique_idx2)}')

        _, coeff1_unique = self.get_invariants_with_axes(
            bs1,
            unique_idx1.tolist(),
            np.repeat(plane_normal[None, :], len(unique_idx1), axis=0),
            verso=1,
            verbose=self.verbose,
        )
        _, coeff2_unique = self.get_invariants_with_axes(
            bs2,
            unique_idx2.tolist(),
            np.repeat((-plane_normal)[None, :], len(unique_idx2), axis=0),
            verso=-1,
            verbose=self.verbose,
        )
        coeff1 = coeff1_unique[inverse_idx1]
        coeff2 = coeff2_unique[inverse_idx2]
        zernike_distance = np.linalg.norm(coeff1 - coeff2, axis=1)
        del coeff1_unique, coeff2_unique, unique_idx1, unique_idx2, inverse_idx1, inverse_idx2  # Free memory

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

        self._log('Building output table')
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
            'PC3_1': pc3_1,
            'plane_u2': sampled_pairs['plane_u2'].to_numpy(dtype=float),
            'plane_v2': sampled_pairs['plane_v2'].to_numpy(dtype=float),
            'PC3_2': pc3_2,
            'rep_x': sampled_pairs['rep_x'].to_numpy(dtype=float),
            'rep_y': sampled_pairs['rep_y'].to_numpy(dtype=float),
            'rep_z': sampled_pairs['rep_z'].to_numpy(dtype=float),
            'scalar_prod': scalar_prod,
            'physical_distance': physical_distance,
            'zernike_distance': zernike_distance,
        })
        del bs1, bs2, coords_bs1, coords_bs2, sampled_pairs, idx1, idx2, physical_distance, pc3_1, pc3_2, normals1, normals2, scalar_prod, coeff1, coeff2  # Free memory

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

            pc3_mean = float((bs1_pc3_mean + bs2_pc3_mean) / 2.0)
            roughness_value = float(np.sqrt((bs1_pc3_var + bs2_pc3_var) / 2.0))
            if summary_type == 'weighted':
                scalar_mean, scalar_unc = weighted_stats(df_out['scalar_prod'].to_numpy(dtype=float), df_out[['plane_u1', 'plane_v1']].to_numpy(dtype=float))
            else:
                scalar_mean, scalar_unc = normal_stats(df_out['scalar_prod'].to_numpy(dtype=float))

            row['gyration_radius'] = gyration_radius
            row['gyration_radius_note'] = gyration_radius_note
            row['flatness'] = flatness
            row['PC3'] = pc3_mean
            row['roughness'] = roughness_value
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
            'PC3',
            'roughness',
            'scalar_prod',
            'scalar_prod_uncertainty',
            'summary_type',
        ])
        summary_df = summary_df[summary_column_order]

        if self.output_name:
            stem = self.output_name
        else:
            stem = f'{self.file_name1}_{self.file_name2}'

        if save_summary:
            summary_path = self.output_path / f'{stem}_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            self._log(f'Summary saved to {summary_path}')

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

            detailed_df = df_out[['idx1', 'res1', 'x1', 'y1', 'z1', 'idx2', 'res2', 'x2', 'y2', 'z2', 'ring_id', 'plane_u1', 'plane_v1', 'PC3_1', 'plane_u2', 'plane_v2', 'PC3_2', 'rep_x', 'rep_y', 'rep_z', 'scalar_prod', 'physical_distance', 'zernike_distance']]

            with open(output_csv, 'w', newline='') as fh:
                for line in meta_lines:
                    fh.write(line + '\n')
                detailed_df.to_csv(fh, index=False)

            self._log(f'Output saved to {output_csv}')

        self._log(f'Plane centroid: {centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}')
        self._log(f'Plane normal: {plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}')
        self._log(f'Projected binding-site centroid on plane: {center_proj[0]:.6f}, {center_proj[1]:.6f}, {center_proj[2]:.6f}')
        self._log(f'Circle radius: {circle_radius:.6f} with 10 rings of width {ring_width:.6f}')

        if plot:
            self._log('Generating plane plots')
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
            self._log(f'Combined subplot saved to {combined_plot}')

        del df_out
        gc.collect()
        return summary_df


def _split_complex_stem(stem):
    """Return (complex_name, chain_tag) from a stem like MY_COMPLEX_A or MY_COMPLEX_1."""
    if '_' not in stem:
        return stem, ''
    complex_name, chain_tag = stem.rsplit('_', 1)
    if len(chain_tag) == 1 and (chain_tag.isalpha() or chain_tag.isdigit()):
        return complex_name, chain_tag
    return stem, ''


def _sort_pair_items(items):
    return sorted(items, key=lambda x: (_split_complex_stem(Path(x).stem)[1], str(x)))


def _discover_pdb_pairs_from_dir(batch_dir):
    batch_dir = Path(batch_dir)
    groups = defaultdict(list)
    for pdb_path in batch_dir.rglob('*.pdb'):
        complex_name, _ = _split_complex_stem(pdb_path.stem)
        groups[complex_name].append(pdb_path)

    tasks = []
    skipped = []
    for complex_name, paths in sorted(groups.items()):
        paths = _sort_pair_items(paths)
        if len(paths) != 2:
            skipped.append((complex_name, len(paths)))
            continue
        tasks.append({
            'source': 'dir',
            'complex_name': complex_name,
            'files': [str(paths[0]), str(paths[1])],
        })
    return tasks, skipped


def _discover_pdb_pairs_from_zip(batch_zip):
    batch_zip = Path(batch_zip)
    groups = defaultdict(list)
    with zipfile.ZipFile(batch_zip, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = info.filename
            if Path(member).suffix.lower() != '.pdb':
                continue
            complex_name, _ = _split_complex_stem(Path(member).stem)
            groups[complex_name].append(member)

    tasks = []
    skipped = []
    for complex_name, members in sorted(groups.items()):
        members = _sort_pair_items(members)
        if len(members) != 2:
            skipped.append((complex_name, len(members)))
            continue
        tasks.append({
            'source': 'zip',
            'zip_path': str(batch_zip),
            'complex_name': complex_name,
            'members': [members[0], members[1]],
        })
    return tasks, skipped


def _expected_summary_rows(sampling_strategy):
    return 2 if sampling_strategy == 'default' else 1


def _prepare_resume_summary(summary_path, expected_rows, force=False):
    """Read the global summary and return completed complex names.

    If a previous run stopped while writing, incomplete rows are removed from the
    active summary and preserved in a .bak file.
    """
    summary_path = Path(summary_path)
    if force or not summary_path.exists() or summary_path.stat().st_size == 0:
        if force and summary_path.exists():
            backup = summary_path.with_suffix(summary_path.suffix + '.force_backup')
            shutil.copy2(summary_path, backup)
            summary_path.unlink()
        return set()

    try:
        df = pd.read_csv(summary_path)
    except Exception:
        backup = summary_path.with_suffix(summary_path.suffix + '.unreadable_backup')
        shutil.copy2(summary_path, backup)
        summary_path.unlink()
        return set()

    if 'complex_name' not in df.columns or 'summary_type' not in df.columns:
        backup = summary_path.with_suffix(summary_path.suffix + '.old_backup')
        shutil.copy2(summary_path, backup)
        summary_path.unlink()
        return set()

    valid = df.dropna(subset=['complex_name', 'summary_type']).copy()
    counts = valid.groupby('complex_name').size()
    completed = set(counts[counts >= expected_rows].index.astype(str))

    cleaned = valid[valid['complex_name'].astype(str).isin(completed)]
    if len(cleaned) != len(df):
        backup = summary_path.with_suffix(summary_path.suffix + '.partial_backup')
        shutil.copy2(summary_path, backup)
        cleaned.to_csv(summary_path, index=False)

    return completed


def _append_summary(summary_path, summary_df):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    summary_df.to_csv(summary_path, mode='a', header=write_header, index=False)


def _write_failure(failure_path, complex_name, error):
    failure_path = Path(failure_path)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failure_path.exists() or failure_path.stat().st_size == 0
    row = pd.DataFrame([{
        'complex_name': complex_name,
        'error': str(error),
    }])
    row.to_csv(failure_path, mode='a', header=write_header, index=False)


def _extract_zip_member(zf, member, destination_dir):
    destination = Path(destination_dir) / Path(member).name
    with zf.open(member, 'r') as src, open(destination, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    return destination


def _run_complex_task(task, options):
    """Worker function. Returns a small summary dataframe or an error."""
    start = time.perf_counter()
    complex_name = task['complex_name']
    verbose = bool(options['verbose'])

    try:
        with _suppress_worker_output(not verbose):
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                if task['source'] == 'zip':
                    with zipfile.ZipFile(task['zip_path'], 'r') as zf:
                        surface1 = _extract_zip_member(zf, task['members'][0], tmpdir)
                        surface2 = _extract_zip_member(zf, task['members'][1], tmpdir)
                else:
                    surface1 = Path(task['files'][0])
                    surface2 = Path(task['files'][1])

                detail_dir = Path(options['output']) / 'details'
                calculator = ComplementaryPlane(
                    surface1,
                    surface2,
                    detail_dir,
                    threshold=options['threshold'],
                    points=options['points'],
                    output_name=complex_name,
                    sampling_strategy=options['sampling_strategy'],
                    verbose=verbose,
                )
                summary_df = calculator.compute(
                    plot=options['plot'],
                    save_csv=options['csv'],
                    save_summary=False,
                )
                summary_df.insert(0, 'complex_name', complex_name)
                summary_df.insert(1, 'protein1_file', Path(surface1).name)
                summary_df.insert(2, 'protein2_file', Path(surface2).name)
                summary_df.insert(3, 'elapsed_seconds', time.perf_counter() - start)
                del calculator
                gc.collect()

        return {
            'ok': True,
            'complex_name': complex_name,
            'summary': summary_df,
            'elapsed_seconds': time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            'ok': False,
            'complex_name': complex_name,
            'error': repr(exc),
            'elapsed_seconds': time.perf_counter() - start,
        }
    finally:
        gc.collect()


def _run_batch(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_zip:
        tasks, skipped = _discover_pdb_pairs_from_zip(args.batch_zip)
    else:
        tasks, skipped = _discover_pdb_pairs_from_dir(args.batch_dir)

    summary_path = Path(args.batch_summary) if args.batch_summary else output_dir / 'batch_summary.csv'
    failure_path = output_dir / 'batch_failures.csv'
    expected_rows = _expected_summary_rows(args.sampling_strategy)
    completed = _prepare_resume_summary(summary_path, expected_rows, force=args.force)

    pending = [task for task in tasks if task['complex_name'] not in completed]

    if args.verbose:
        print(f'Found {len(tasks)} complete complexes.')
        if skipped:
            print(f'Skipped {len(skipped)} incomplete groups (not exactly 2 PDB files).')
        print(f'Already completed in summary: {len(completed)}')
        print(f'Pending complexes: {len(pending)}')
        print(f'Global summary: {summary_path}')

    if not pending:
        if args.verbose:
            print(f'Nothing to do. Summary already contains {len(completed)} completed complexes: {summary_path}')
        return

    worker_options = {
        'output': str(output_dir),
        'threshold': args.threshold,
        'points': args.points,
        'sampling_strategy': args.sampling_strategy,
        'plot': args.plot,
        'csv': args.csv,
        'verbose': args.verbose,
    }

    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    progress = tqdm.tqdm(
        total=len(pending),
        desc='Complexes',
        unit='complex',
        disable=args.verbose,
        bar_format=bar_format,
    )

    try:
        if args.workers == 1:
            for task in pending:
                result = _run_complex_task(task, worker_options)
                if result['ok']:
                    _append_summary(summary_path, result['summary'])
                    if args.verbose:
                        print(f"OK {result['complex_name']} ({result['elapsed_seconds']:.1f} s)")
                else:
                    _write_failure(failure_path, result['complex_name'], result['error'])
                    if args.verbose:
                        print(f"FAILED {result['complex_name']}: {result['error']}")
                progress.update(1)
                gc.collect()
        else:
            # max_tasks_per_child=1 is slower, but it gives the cleanest RAM reset:
            # each complex runs in a fresh process, then the process exits.
            try:
                executor = ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1)
            except TypeError:
                executor = ProcessPoolExecutor(max_workers=args.workers)

            futures = {}
            try:
                futures = {executor.submit(_run_complex_task, task, worker_options): task for task in pending}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'ok': False, 'complex_name': task['complex_name'], 'error': repr(exc)}

                    if result['ok']:
                        _append_summary(summary_path, result['summary'])
                        if args.verbose:
                            print(f"OK {result['complex_name']} ({result.get('elapsed_seconds', float('nan')):.1f} s)")
                    else:
                        _write_failure(failure_path, result['complex_name'], result['error'])
                        if args.verbose:
                            print(f"FAILED {result['complex_name']}: {result['error']}")
                    progress.update(1)
            finally:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True)
    finally:
        progress.close()
        gc.collect()

    if args.verbose:
        print(f'Done. Global summary saved to {summary_path}')
        if failure_path.exists() and failure_path.stat().st_size > 0:
            print(f'Failures, if any, were saved to {failure_path}')


def main():
    args = build_cli_complementary_plane2()

    if getattr(args, 'batch_dir', None) or getattr(args, 'batch_zip', None):
        _run_batch(args)
        return

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
    calculator.compute(
        plot=args.plot,
        save_csv=getattr(args, 'csv', False),
        save_summary=True,
    )


if __name__ == '__main__':
    # Required for ProcessPoolExecutor on Windows/macOS spawn mode.
    main()
