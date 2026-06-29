#!/usr/bin/env python3
"""Compute binding-site radius for each paired PDB complex in a zip archive.

The script reproduces the radius calculation used by the complementary-plane
workflow up to the concentric-ring construction step, then writes a CSV with
only the complex name and the resulting radius.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tqdm

from binding_site_utils import get_binding_site_mask
from plane_geometry import build_concentric_rings, fit_plane, project_point_to_plane, project_surface_to_plane
from surface_processing import load_surface_input


def _split_complex_stem(stem):
    if '_' not in stem:
        return stem, ''
    complex_name, chain_tag = stem.rsplit('_', 1)
    if chain_tag and chain_tag.isalnum():
        return complex_name, chain_tag
    return stem, ''


def _sort_pair_items(items):
    return sorted(items, key=lambda item: (_split_complex_stem(Path(item).stem)[1], str(item)))


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
            'zip_path': str(batch_zip),
            'complex_name': complex_name,
            'members': [members[0], members[1]],
        })
    return tasks, skipped


def _extract_zip_member(zf, member, destination_dir, index):
    destination = Path(destination_dir) / f'{index}_{Path(member).name}'
    with zf.open(member, 'r') as src, open(destination, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    return destination


def _compute_radius_for_pair(surface1_path, surface2_path, threshold):
    surface1, _ = load_surface_input(surface1_path)
    surface2, _ = load_surface_input(surface2_path)

    coords1 = surface1[['x', 'y', 'z']].to_numpy(dtype=float)
    coords2 = surface2[['x', 'y', 'z']].to_numpy(dtype=float)

    mask1, _ = get_binding_site_mask(coords1, coords2, threshold)
    mask2, _ = get_binding_site_mask(coords2, coords1, threshold)

    bs1 = surface1[mask1 == 1].reset_index(drop=True)
    bs2 = surface2[mask2 == 1].reset_index(drop=True)
    if len(bs1) < 1 or len(bs2) < 1:
        raise ValueError('Binding site extraction produced an empty surface')

    coords_bs1 = bs1[['x', 'y', 'z']].to_numpy(dtype=float)
    coords_bs2 = bs2[['x', 'y', 'z']].to_numpy(dtype=float)
    binding_site_centroid = np.vstack((coords_bs1, coords_bs2)).mean(axis=0)

    distances_12 = np.linalg.norm(coords_bs1[:, None, :] - coords_bs2[None, :, :], axis=2)
    nearest_idx2 = np.argmin(distances_12, axis=1)
    paired_coords2 = coords_bs2[nearest_idx2]
    midpoints1 = (coords_bs1 + paired_coords2) / 2.0

    nearest_idx1 = np.argmin(distances_12, axis=0)
    paired_coords1 = coords_bs1[nearest_idx1]
    midpoints2 = (paired_coords1 + coords_bs2) / 2.0

    midpoints = np.vstack((midpoints1, midpoints2))
    centroid, basis, _, _, _ = fit_plane(midpoints)
    center_uv, _ = project_point_to_plane(binding_site_centroid, centroid, basis)
    _, plane_coords1, _ = project_surface_to_plane(bs1, centroid, basis)
    _, plane_coords2, _ = project_surface_to_plane(bs2, centroid, basis)

    circle_radius, _, _, _, _, _ = build_concentric_rings(
        plane_coords1,
        plane_coords2,
        center_uv,
        n_rings=10,
        min_outer_points=10,
    )

    return float(circle_radius)


def _run_complex_task(task, threshold):
    complex_name = task['complex_name']
    try:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(task['zip_path'], 'r') as zf:
                surface1 = _extract_zip_member(zf, task['members'][0], tmpdir, 1)
                surface2 = _extract_zip_member(zf, task['members'][1], tmpdir, 2)

            radius = _compute_radius_for_pair(surface1, surface2, threshold)
        return {'complex_name': complex_name, 'radius': radius}
    except Exception:
        return {'complex_name': complex_name, 'radius': float('nan')}


def build_parser():
    parser = argparse.ArgumentParser(
        description='Compute the outer binding-site radius for each paired PDB complex inside a zip archive.'
    )
    parser.add_argument('--batch-zip', required=True, help='zip archive containing paired PDB files')
    parser.add_argument('-o', '--output', required=True, help='output CSV path')
    parser.add_argument('-t', '--threshold', type=float, default=5.0, help='binding-site distance threshold (default: 5.0)')
    parser.add_argument('--workers', type=int, default=1, help='number of worker processes (default: 1)')
    return parser


def main():
    args = build_parser().parse_args()
    tasks, skipped = _discover_pdb_pairs_from_zip(args.batch_zip)

    results = []
    progress = tqdm.tqdm(total=len(tasks), desc='Complexes', unit='complex')
    if args.workers == 1:
        for task in tasks:
            results.append(_run_complex_task(task, args.threshold))
            progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_run_complex_task, task, args.threshold): task for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())
                progress.update(1)

    progress.close()

    results = sorted(results, key=lambda row: row['complex_name'])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['complex_name', 'radius'])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()