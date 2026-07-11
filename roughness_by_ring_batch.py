#!/usr/bin/env python3
"""Batch roughness analysis per ring for zip archives or plot-only CSV comparisons.

For each complex the script rebuilds the complementary plane workflow with the
default sampling strategy only, then computes a roughness value for every ring
instead of on the whole binding site.

If two zip files are provided, it also generates a comparison plot with a
standard ring-ID view and a real-radius view.
"""

from __future__ import annotations

import argparse
import gc
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import tqdm

from binding_site_utils import get_binding_site_mask
from get_complementary_plane import (
    _append_summary,
    _discover_pdb_pairs_from_zip,
    _extract_zip_member,
    _suppress_worker_output,
    _stable_seed_from_parts,
    _write_failure,
)
from plane_geometry import (
    build_concentric_rings,
    fit_plane,
    project_point_to_plane,
    project_surface_to_plane,
    select_ring_pairs,
)
from surface_processing import load_surface_input


N_RINGS = 10


class RoughnessByRingCalculator:
    def __init__(self, surface_file1, surface_file2, output_path, threshold=5.0, points=100, verbose=False):
        self.surface_file1 = Path(surface_file1)
        self.surface_file2 = Path(surface_file2)
        self.file_name1 = self.surface_file1.stem
        self.file_name2 = self.surface_file2.stem
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.points = points
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

    def compute(self):
        self._log(f'Loading surfaces: {self.file_name1} and {self.file_name2}')
        self._log(f'Extracting binding sites with threshold {self.threshold} Å')

        bs1, bs2 = self.get_binding_sites(self.surface1, self.surface2, self.threshold)
        if len(bs1) < 1 or len(bs2) < 1:
            raise ValueError('Binding site extraction produced an empty surface')

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

        tree_bs2 = cKDTree(coords_bs2)
        _, nearest_idx2 = tree_bs2.query(coords_bs1, k=1)
        paired_coords2 = coords_bs2[nearest_idx2]
        midpoints1 = (coords_bs1 + paired_coords2) / 2.0

        tree_bs1 = cKDTree(coords_bs1)
        _, nearest_idx1 = tree_bs1.query(coords_bs2, k=1)
        paired_coords1 = coords_bs1[nearest_idx1]
        midpoints2 = (paired_coords1 + coords_bs2) / 2.0
        del tree_bs1, tree_bs2, nearest_idx1, nearest_idx2, paired_coords1, paired_coords2

        midpoints = np.vstack((midpoints1, midpoints2))
        self._log(f'Collected midpoints: bs1->bs2={len(midpoints1)}, bs2->bs1={len(midpoints2)}, total={len(midpoints)}')
        del midpoints1, midpoints2

        centroid, basis, _, _, _ = fit_plane(midpoints)
        del midpoints
        plane_normal = basis[2]
        center_uv, _ = project_point_to_plane(binding_site_centroid, centroid, basis)
        _, plane_coords1, _ = project_surface_to_plane(bs1, centroid, basis)
        _, plane_coords2, _ = project_surface_to_plane(bs2, centroid, basis)

        self._log('Projecting the binding-site centroid on the plane and building concentric rings')
        circle_radius, ring_width, _, _, ring_ids1, ring_ids2 = build_concentric_rings(
            plane_coords1,
            plane_coords2,
            center_uv,
            n_rings=N_RINGS,
            min_outer_points=10,
        )
        self._log(f'Circle center projected at u={center_uv[0]:.6f}, v={center_uv[1]:.6f}')
        self._log(f'Final circle radius={circle_radius:.6f}; ring width={ring_width:.6f}')

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
            n_rings=N_RINGS,
            random_state=_stable_seed_from_parts(self.file_name1, self.file_name2, self.threshold, self.points, 'default'),
        )

        if sampled_pairs.empty:
            raise ValueError('Ring-based sampling produced no matched pairs')

        idx1 = sampled_pairs['idx1'].to_numpy(dtype=int)
        idx2 = sampled_pairs['idx2'].to_numpy(dtype=int)

        pc3_1 = (coords_bs1[idx1] - centroid) @ plane_normal
        pc3_2 = (coords_bs2[idx2] - centroid) @ plane_normal

        row = {'radius': float(circle_radius)}
        sampled_ring_ids = sampled_pairs['ring_id'].to_numpy(dtype=int)
        for rid in range(1, N_RINGS + 1):
            ring_mask = sampled_ring_ids == rid
            if not np.any(ring_mask):
                row[f'roughness_ring{rid}'] = float('nan')
                continue

            variance1 = float(np.var(pc3_1[ring_mask]))
            variance2 = float(np.var(pc3_2[ring_mask]))
            row[f'roughness_ring{rid}'] = float(np.sqrt((variance1 + variance2) / 2.0))

        summary_df = pd.DataFrame([row])
        summary_df.insert(0, 'complex_name', f'{self.file_name1}_{self.file_name2}')

        self._log(f'Plane centroid: {centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}')
        self._log(f'Plane normal: {plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}')
        self._log(f'Circle radius: {circle_radius:.6f} with {N_RINGS} rings of width {ring_width:.6f}')

        del bs1, bs2, coords_bs1, coords_bs2, sampled_pairs, idx1, idx2, pc3_1, pc3_2, ring_ids1, ring_ids2
        gc.collect()
        return summary_df


def _expected_output_paths(zip_path, output_dir):
    zip_path = Path(zip_path)
    stem = zip_path.stem
    output_dir = Path(output_dir)
    summary_path = output_dir / f'{stem}_roughness_summary.csv'
    failure_path = output_dir / f'{stem}_roughness_failures.csv'
    return summary_path, failure_path


def _prepare_resume_summary_roughness(summary_path, force=False):
    summary_path = Path(summary_path)
    if force or not summary_path.exists() or summary_path.stat().st_size == 0:
        if force and summary_path.exists():
            backup = summary_path.with_suffix(summary_path.suffix + '.force_backup')
            summary_path.replace(backup)
        return set()

    try:
        df = pd.read_csv(summary_path)
    except Exception:
        backup = summary_path.with_suffix(summary_path.suffix + '.unreadable_backup')
        summary_path.replace(backup)
        return set()

    if 'complex_name' not in df.columns:
        backup = summary_path.with_suffix(summary_path.suffix + '.old_backup')
        summary_path.replace(backup)
        return set()

    valid = df.dropna(subset=['complex_name']).copy()
    completed = set(valid['complex_name'].astype(str).unique())

    if len(valid) != len(df):
        backup = summary_path.with_suffix(summary_path.suffix + '.partial_backup')
        summary_path.replace(backup)
        valid.to_csv(summary_path, index=False)

    return completed


def _run_zip_task(task, options):
    start = time.perf_counter()
    zip_path = Path(task['zip_path'])
    verbose = bool(options['verbose'])

    try:
        with _suppress_worker_output(not verbose):
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                detail_dir = Path(options['output']) / 'details'
                summary_rows = []

                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for complex_task in task['tasks']:
                        complex_start = time.perf_counter()
                        surface1 = _extract_zip_member(zf, complex_task['members'][0], tmpdir)
                        surface2 = _extract_zip_member(zf, complex_task['members'][1], tmpdir)

                        calculator = RoughnessByRingCalculator(
                            surface1,
                            surface2,
                            detail_dir,
                            threshold=options['threshold'],
                            points=options['points'],
                            verbose=verbose,
                        )
                        summary_df = calculator.compute()
                        summary_rows.append(summary_df)
                        _ = time.perf_counter() - complex_start
                        del calculator
                        gc.collect()

                if summary_rows:
                    summary_df = pd.concat(summary_rows, ignore_index=True)
                else:
                    summary_df = pd.DataFrame()

        return {
            'ok': True,
            'zip_path': str(zip_path),
            'summary': summary_df,
            'elapsed_seconds': time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            'ok': False,
            'zip_path': str(zip_path),
            'error': repr(exc),
            'elapsed_seconds': time.perf_counter() - start,
        }
    finally:
        gc.collect()


def _build_zip_task(zip_path):
    tasks, skipped = _discover_pdb_pairs_from_zip(zip_path)
    return {
        'zip_path': str(zip_path),
        'tasks': tasks,
        'skipped': skipped,
    }


def _load_summary_for_plot(summary_path):
    df = pd.read_csv(summary_path)
    ring_columns = [column for column in df.columns if column.startswith('roughness_ring')]
    if not ring_columns:
        raise ValueError(f'No roughness ring columns found in {summary_path}')
    return df


def _summary_display_name(summary_path):
    name = Path(summary_path).name
    if name.endswith('_roughness_summary.csv'):
        return name[:-len('_roughness_summary.csv')]
    return Path(name).stem


def _normalize_complex_key(complex_name):
    return str(complex_name).strip().upper()[:4]


def _common_complex_names(df_a, df_b):
    if 'complex_name' not in df_a.columns or 'complex_name' not in df_b.columns:
        raise ValueError('complex_name column is required to compute common complexes')

    keys_a = {}
    for complex_name in df_a['complex_name'].dropna().astype(str).tolist():
        key = _normalize_complex_key(complex_name)
        keys_a.setdefault(key, set()).add(complex_name)

    keys_b = {}
    for complex_name in df_b['complex_name'].dropna().astype(str).tolist():
        key = _normalize_complex_key(complex_name)
        keys_b.setdefault(key, set()).add(complex_name)

    common_keys = sorted(set(keys_a) & set(keys_b))
    names_a = sorted({name for key in common_keys for name in keys_a[key]})
    names_b = sorted({name for key in common_keys for name in keys_b[key]})
    return names_a, names_b


def _mean_and_sample_uncertainty(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float('nan'), float('nan')
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return mean_value, 0.0
    return mean_value, float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _aggregate_roughness_by_ring(df_summary):
    if 'complex_name' not in df_summary.columns:
        raise ValueError('complex_name column is required to aggregate roughness')

    ring_ids = []
    for column in df_summary.columns:
        if column.startswith('roughness_ring'):
            ring_ids.append(int(column.replace('roughness_ring', '')))
    ring_ids = sorted(set(ring_ids))

    per_ring_values = {rid: [] for rid in ring_ids}
    for _, complex_group in df_summary.groupby('complex_name', sort=False):
        for rid in ring_ids:
            column = f'roughness_ring{rid}'
            values = pd.to_numeric(complex_group[column], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) > 0:
                per_ring_values[rid].append(float(values[0]))

    aggregated = {}
    for rid in ring_ids:
        mean_value, sample_uncertainty = _mean_and_sample_uncertainty(per_ring_values[rid])
        aggregated[rid] = {
            'mean': mean_value,
            'uncertainty': sample_uncertainty,
            'count': int(len(per_ring_values[rid])),
        }
    return ring_ids, aggregated


def _aggregate_real_roughness_by_ring(df_summary, bin_width=2.0, max_x=50.0):
    if 'complex_name' not in df_summary.columns:
        raise ValueError('complex_name column is required to aggregate roughness')
    if 'radius' not in df_summary.columns:
        raise ValueError('radius column is required to compute real-valued ring positions')

    ring_ids = []
    for column in df_summary.columns:
        if column.startswith('roughness_ring'):
            ring_ids.append(int(column.replace('roughness_ring', '')))
    ring_ids = sorted(set(ring_ids))

    bin_edges = np.arange(0.0, max_x + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2.0
    bin_ids = list(range(len(bin_centers)))
    per_bin_values = {bid: [] for bid in bin_ids}

    for _, complex_group in df_summary.groupby('complex_name', sort=False):
        radius_values = pd.to_numeric(complex_group['radius'], errors='coerce').to_numpy(dtype=float)
        radius_values = radius_values[np.isfinite(radius_values)]
        if len(radius_values) == 0:
            continue

        radius = float(radius_values[0])
        if not np.isfinite(radius) or radius <= 0:
            continue

        for rid in ring_ids:
            column = f'roughness_ring{rid}'
            values = pd.to_numeric(complex_group[column], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue

            x_position = (rid - 0.5) * radius / float(N_RINGS)
            if x_position > max_x:
                continue

            bin_index = int(np.floor(x_position / bin_width))
            bin_index = min(bin_index, len(bin_ids) - 1)
            per_bin_values[bin_index].append(float(values[0]))

    aggregated = {}
    for bid in bin_ids:
        mean_value, sample_uncertainty = _mean_and_sample_uncertainty(per_bin_values[bid])
        aggregated[bid] = {
            'mean': mean_value,
            'uncertainty': sample_uncertainty,
            'count': int(len(per_bin_values[bid])),
        }

    return bin_edges, bin_centers, aggregated


def _plot_standard_panel(ax, datasets, ring_ids, labels):
    from matplotlib.lines import Line2D

    x_positions = np.arange(len(ring_ids))
    offset_step = 0.18 / max(len(datasets), 1)
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['s', 'o']

    for dataset_index, ring_data in enumerate(datasets):
        offset = (dataset_index - (len(datasets) - 1) / 2.0) * offset_step
        x_pos = x_positions + offset
        means = np.array([ring_data[rid]['mean'] for rid in ring_ids], dtype=float)
        uncertainties = np.array([ring_data[rid]['uncertainty'] for rid in ring_ids], dtype=float)
        color = colors[dataset_index % len(colors)]
        marker = markers[dataset_index % len(markers)]

        ax.errorbar(
            x_pos,
            means,
            yerr=uncertainties,
            fmt='none',
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            capthick=1.4,
            zorder=2,
        )
        ax.scatter(
            x_pos,
            means,
            s=22,
            marker=marker,
            facecolors=color,
            edgecolors='black',
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlabel('Ring ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Roughness (Å)', fontsize=12, fontweight='bold')
    ax.set_title('Standard ring comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{ring_id}' for ring_id in ring_ids])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if len(labels) > 1:
        handles = []
        for dataset_index, label in enumerate(labels):
            color = colors[dataset_index % len(colors)]
            marker = markers[dataset_index % len(markers)]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle='none',
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8,
                    label=label,
                )
            )
        ax.legend(handles=handles, fontsize=10, loc='best', title='dataset')


def _plot_real_panel(ax, datasets, bin_centers, labels):
    from matplotlib.lines import Line2D

    x_positions = np.asarray(bin_centers, dtype=float)
    offset_step = 0.18 / max(len(datasets), 1)
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['s', 'o']

    for dataset_index, bin_data in enumerate(datasets):
        offset = (dataset_index - (len(datasets) - 1) / 2.0) * offset_step
        x_pos = x_positions + offset
        means = np.array([bin_data[bid]['mean'] for bid in range(len(bin_centers))], dtype=float)
        uncertainties = np.array([bin_data[bid]['uncertainty'] for bid in range(len(bin_centers))], dtype=float)
        color = colors[dataset_index % len(colors)]
        marker = markers[dataset_index % len(markers)]

        ax.errorbar(
            x_pos,
            means,
            yerr=uncertainties,
            fmt='none',
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            capthick=1.4,
            zorder=2,
        )
        ax.scatter(
            x_pos,
            means,
            s=22,
            marker=marker,
            facecolors=color,
            edgecolors='black',
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlabel('Real radial distance bin center (Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Roughness (Å)', fontsize=12, fontweight='bold')
    ax.set_title('Real-radius comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f'{center:.0f}' for center in bin_centers], rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if len(labels) > 1:
        handles = []
        for dataset_index, label in enumerate(labels):
            color = colors[dataset_index % len(colors)]
            marker = markers[dataset_index % len(markers)]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle='none',
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8,
                    label=label,
                )
            )
        ax.legend(handles=handles, fontsize=10, loc='best', title='dataset')


def _plot_comparison(summary_paths, output_dir, in_common=False):
    import matplotlib.pyplot as plt

    if len(summary_paths) != 2:
        return None

    df_summaries = [_load_summary_for_plot(summary_path) for summary_path in summary_paths]
    labels = [_summary_display_name(summary_path) for summary_path in summary_paths]

    if in_common:
        common_a, common_b = _common_complex_names(df_summaries[0], df_summaries[1])
        if not common_a or not common_b:
            raise ValueError('No common complexes found for --in-common')
        df_summaries = [
            df_summaries[0][df_summaries[0]['complex_name'].astype(str).isin(common_a)].copy(),
            df_summaries[1][df_summaries[1]['complex_name'].astype(str).isin(common_b)].copy(),
        ]

    ring_sets = [set(_aggregate_roughness_by_ring(df_summary)[0]) for df_summary in df_summaries]
    common_ring_ids = sorted(set.intersection(*ring_sets))
    if not common_ring_ids:
        raise ValueError('No common roughness ring columns found between the two summaries')

    standard_data = []
    for df_summary in df_summaries:
        _, aggregated = _aggregate_roughness_by_ring(df_summary)
        standard_data.append({rid: aggregated[rid] for rid in common_ring_ids})

    real_data = []
    bin_centers = None
    for df_summary in df_summaries:
        _, current_bin_centers, aggregated = _aggregate_real_roughness_by_ring(df_summary)
        if bin_centers is None:
            bin_centers = current_bin_centers
        real_data.append(aggregated)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    _plot_standard_panel(axes[0], standard_data, common_ring_ids, labels)
    _plot_real_panel(axes[1], real_data, bin_centers, labels)

    fig.suptitle(f'Roughness comparison ({" vs ".join(labels)})', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()

    output_path = Path(output_dir) / f'{labels[0]}_vs_{labels[1]}_roughness_comparison.pdf'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _run_batch_on_zip(zip_path, args):
    zip_task = _build_zip_task(zip_path)
    summary_path, failure_path = _expected_output_paths(zip_path, args.output)
    expected_rows = 1
    completed = _prepare_resume_summary_roughness(summary_path, force=args.force)
    pending = [task for task in zip_task['tasks'] if task['complex_name'] not in completed]

    if args.verbose:
        print(f'Found {len(zip_task["tasks"])} complete complexes in {zip_path}')
        if zip_task['skipped']:
            print(f'Skipped {len(zip_task["skipped"])} incomplete groups (not exactly 2 PDB files).')
        print(f'Already completed in summary: {len(completed)}')
        print(f'Pending complexes: {len(pending)}')
        print(f'Global summary: {summary_path}')

    if not pending:
        if args.verbose:
            print(f'Nothing to do. Summary already contains {len(completed)} completed complexes: {summary_path}')
        return summary_path

    worker_options = {
        'output': str(args.output),
        'threshold': args.threshold,
        'points': args.points,
        'verbose': args.verbose,
    }

    progress = tqdm.tqdm(
        total=len(pending),
        desc=f'Complexes ({Path(zip_path).name})',
        unit='complex',
        disable=args.verbose,
    )

    try:
        if args.workers == 1:
            for task in pending:
                result = _run_zip_task({'zip_path': str(zip_path), 'tasks': [task]}, worker_options)
                if result['ok']:
                    _append_summary(summary_path, result['summary'])
                    if args.verbose:
                        print(f"OK {task['complex_name']} ({result['elapsed_seconds']:.1f} s)")
                else:
                    _write_failure(failure_path, task['complex_name'], result['error'])
                    if args.verbose:
                        print(f"FAILED {task['complex_name']}: {result['error']}")
                progress.update(1)
                gc.collect()
        else:
            try:
                executor = ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1)
            except TypeError:
                executor = ProcessPoolExecutor(max_workers=args.workers)

            futures = {}
            try:
                for task in pending:
                    futures[executor.submit(_run_zip_task, {'zip_path': str(zip_path), 'tasks': [task]}, worker_options)] = task

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'ok': False, 'error': repr(exc)}

                    if result['ok']:
                        _append_summary(summary_path, result['summary'])
                        if args.verbose:
                            print(f"OK {task['complex_name']} ({result.get('elapsed_seconds', float('nan')):.1f} s)")
                    else:
                        _write_failure(failure_path, task['complex_name'], result['error'])
                        if args.verbose:
                            print(f"FAILED {task['complex_name']}: {result['error']}")
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

    return summary_path


def build_cli():
    parser = argparse.ArgumentParser(
        description='Batch roughness-per-ring analysis for zip archives or plot-only CSV comparisons.'
    )
    parser.add_argument('--batch-zip', help='First zip archive with paired PDB files')
    parser.add_argument('--compare-batch-zip', help='Second zip archive for comparison plots')
    parser.add_argument('--summary-csv', help='First summary CSV for plot-only mode')
    parser.add_argument('--compare-summary-csv', help='Second summary CSV for plot-only mode')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-t', '--threshold', type=float, default=5.0, help='Distance threshold in angstroms')
    parser.add_argument('-n', '--points', type=int, default=100, help='Number of points per ring')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for batch mode')
    parser.add_argument('--force', action='store_true', help='Ignore existing summaries and recompute all complexes')
    parser.add_argument('--in-common', action='store_true', help='Plot only complexes common to both inputs, matched by the first 4 characters of complex_name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()


def main():
    args = build_cli()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = []
    batch_mode = bool(args.batch_zip or args.compare_batch_zip)
    csv_mode = bool(args.summary_csv or args.compare_summary_csv)

    if batch_mode and csv_mode:
        raise SystemExit('Use either zip inputs or CSV inputs, not both.')

    if csv_mode:
        if not args.summary_csv or not args.compare_summary_csv:
            raise SystemExit('CSV plot-only mode requires both --summary-csv and --compare-summary-csv.')
        summary_paths = [Path(args.summary_csv), Path(args.compare_summary_csv)]
    else:
        if not args.batch_zip:
            raise SystemExit('Batch mode requires --batch-zip.')
        summary_paths.append(_run_batch_on_zip(args.batch_zip, args))
        if args.compare_batch_zip:
            summary_paths.append(_run_batch_on_zip(args.compare_batch_zip, args))

    if len(summary_paths) == 2:
        plot_path = _plot_comparison(summary_paths, output_dir, in_common=args.in_common)
        if args.verbose:
            print(f'Comparison plot saved to {plot_path}')


if __name__ == '__main__':
    main()