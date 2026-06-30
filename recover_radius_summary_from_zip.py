#!/usr/bin/env python3
"""
Recover only the complementary-plane outer radius from a ZIP of paired PDB complexes.

This is a single-file, radius-only version of the complementary-plane workflow:
- input: one ZIP archive containing paired PDB files named like COMPLEX_A.pdb / COMPLEX_B.pdb
- output: one CSV with exactly: complex_name,radius

It intentionally stops before any sampling/Zernike calculation. In the original workflow,
the radius is computed before the sampling-strategy block, so the radius does not depend on
sampling_strategy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


np.seterr(divide="ignore", invalid="ignore")


# -----------------------------
# ZIP / pair discovery utilities
# -----------------------------

def split_complex_stem(stem: str) -> Tuple[str, str]:
    """Return (complex_name, chain_tag) from stems like MY_COMPLEX_A or MY_COMPLEX_1."""
    if "_" not in stem:
        return stem, ""
    complex_name, chain_tag = stem.rsplit("_", 1)
    if chain_tag and chain_tag.isalnum():
        return complex_name, chain_tag
    return stem, ""


def sort_pair_items(items: Sequence[str]) -> List[str]:
    return sorted(items, key=lambda x: (split_complex_stem(Path(x).stem)[1], str(x)))


def discover_pdb_pairs_from_zip(batch_zip: Path) -> Tuple[List[dict], List[Tuple[str, int]]]:
    """Find groups with exactly two .pdb files inside the zip."""
    groups: Dict[str, List[str]] = defaultdict(list)

    with zipfile.ZipFile(batch_zip, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = info.filename
            if Path(member).suffix.lower() != ".pdb":
                continue
            complex_name, _ = split_complex_stem(Path(member).stem)
            groups[complex_name].append(member)

    tasks: List[dict] = []
    skipped: List[Tuple[str, int]] = []
    for complex_name, members in sorted(groups.items()):
        members = sort_pair_items(members)
        if len(members) != 2:
            skipped.append((complex_name, len(members)))
            continue
        tasks.append(
            {
                "zip_path": str(batch_zip),
                "complex_name": complex_name,
                "members": [members[0], members[1]],
            }
        )
    return tasks, skipped


def extract_zip_member(zf: zipfile.ZipFile, member: str, destination_dir: Path) -> Path:
    """Extract a single zip member into destination_dir using only its basename."""
    destination = destination_dir / Path(member).name
    with zf.open(member, "r") as src, open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return destination


# -----------------------------
# Surface generation/loading
# -----------------------------

def range_char(start: str, stop: str) -> Iterable[str]:
    return (chr(n) for n in range(ord(start), ord(stop) + 1))


def parse_dms_surface(dms_path: Path) -> pd.DataFrame:
    """Parse a DMS molecular surface file into columns res,x,y,z,nx,ny,nz."""
    colnames = [character for character in range_char("A", "K")]

    dms_surf = pd.read_csv(
        dms_path,
        names=colnames,
        header=None,
        delimiter=r"\s+",
        usecols=list(range(len(colnames))),
    )

    # Clean residue identifier columns only; keep numeric columns untouched.
    dms_surf[["A", "B", "C"]] = dms_surf[["A", "B", "C"]].astype(str).replace(
        r"[^\w\s]|_", "", regex=True
    )

    df = (
        dms_surf.assign(
            res=np.where(
                dms_surf["B"].astype(str).str[-1].str.isnumeric(),
                dms_surf["A"].astype(str) + "_" + dms_surf["B"].astype(str) + "_" + dms_surf["C"],
                dms_surf["A"].astype(str)
                + "_"
                + (
                    dms_surf["B"].astype(str)
                    .str.extract(r"(\d+\.?\d*)([A-Za-z]*)", expand=True)
                    .agg("_".join, axis=1)
                )
                + "_"
                + dms_surf["C"],
            )
        )
        .query('G.str[0] == "S"')
        .filter(items=["res", "D", "E", "F", "I", "J", "K"])
        .set_axis(["res", "x", "y", "z", "nx", "ny", "nz"], axis=1)
        .reset_index(drop=True)
    )

    for col in ["x", "y", "z", "nx", "ny", "nz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y", "z", "nx", "ny", "nz"]).reset_index(drop=True)
    return df


def load_pdb_surface(pdb_path: Path, dms_bin: str = "dms") -> pd.DataFrame:
    """Run DMS on a PDB and load the resulting surface."""
    with TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dms_output = tmpdir / f"{pdb_path.stem}.dms"
        run_dms = [dms_bin, str(pdb_path), "-n", "-o", str(dms_output)]
        subprocess.run(run_dms, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # Same formatting fix used in the previous workflow.
        with open(dms_output, "r") as file:
            file_lines = [f"{x[:14]} {x[14:]}" for x in file.readlines()]
        with open(dms_output, "w") as file:
            file.writelines(file_lines)

        surface = parse_dms_surface(dms_output)

    if surface.empty:
        raise ValueError(f"DMS produced an empty surface for {pdb_path.name}")
    return surface


# -----------------------------
# Geometry copied from the original radius path
# -----------------------------

def get_binding_site_mask(coords1: np.ndarray, coords2: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    tree = cKDTree(coords2)
    min_dist, _ = tree.query(coords1, k=1)
    return (min_dist <= threshold).astype(int)


def get_binding_sites(surface1: pd.DataFrame, surface2: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coords1 = surface1[["x", "y", "z"]].to_numpy(dtype=float)
    coords2 = surface2[["x", "y", "z"]].to_numpy(dtype=float)

    mask1 = get_binding_site_mask(coords1, coords2, threshold)
    mask2 = get_binding_site_mask(coords2, coords1, threshold)

    bs1 = surface1[mask1 == 1].reset_index(drop=True)
    bs2 = surface2[mask2 == 1].reset_index(drop=True)
    if len(bs1) < 1 or len(bs2) < 1:
        raise ValueError("Binding site extraction produced an empty surface")
    return bs1, bs2


def fit_plane(midpoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    midpoints = np.asarray(midpoints, dtype=float)
    if len(midpoints) < 3:
        raise ValueError("At least 3 midpoint points are required to fit a plane")

    pca = PCA(n_components=3)
    pca.fit(midpoints)
    centroid = pca.mean_
    basis = pca.components_
    return centroid, basis


def project_point_to_plane(point: np.ndarray, centroid: np.ndarray, basis: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    centered = point - centroid
    plane_u = float(centered @ basis[0])
    plane_v = float(centered @ basis[1])
    return np.array([plane_u, plane_v], dtype=float)


def project_coords_to_plane(coords: np.ndarray, centroid: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    centered = coords - centroid
    plane_u = centered @ basis[0]
    plane_v = centered @ basis[1]
    return np.column_stack((plane_u, plane_v))


def build_concentric_rings(
    plane_coords1: np.ndarray,
    plane_coords2: np.ndarray,
    center_uv: np.ndarray,
    n_rings: int = 10,
    min_outer_points: int = 10,
) -> Tuple[float, float]:
    """Return (outer_radius, ring_width) using the same adaptive rule as the original workflow."""
    center_uv = np.asarray(center_uv, dtype=float)
    radii1 = np.linalg.norm(plane_coords1 - center_uv, axis=1)
    radii2 = np.linalg.norm(plane_coords2 - center_uv, axis=1)

    max_radius = float(max(np.max(radii1), np.max(radii2)))
    if np.isclose(max_radius, 0.0):
        raise ValueError("Projected binding sites are degenerate in the complementary plane")

    total_points1 = len(radii1)
    total_points2 = len(radii2)
    min_total = min(total_points1, total_points2)

    if min_total < 50:
        adaptive_min_outer_points = max(1, min_total // 10)
    elif min_total < 100:
        adaptive_min_outer_points = max(2, min_outer_points // 2)
    else:
        adaptive_min_outer_points = min_outer_points
    adaptive_min_outer_points = int(adaptive_min_outer_points)

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
        if outer_count1 >= adaptive_min_outer_points and outer_count2 >= adaptive_min_outer_points:
            return float(radius), float(ring_width)

        radius *= 0.95

    raise ValueError(
        f"Unable to find a circle radius where the outer ring contains at least "
        f"{adaptive_min_outer_points} points for both binding sites"
    )


def compute_radius_for_pair(surface_file1: Path, surface_file2: Path, threshold: float, dms_bin: str) -> float:
    """Compute only the outer complementary-plane radius for one PDB pair."""
    surface1 = load_pdb_surface(surface_file1, dms_bin=dms_bin)
    surface2 = load_pdb_surface(surface_file2, dms_bin=dms_bin)

    bs1, bs2 = get_binding_sites(surface1, surface2, threshold)
    del surface1, surface2
    gc.collect()

    coords_bs1 = bs1[["x", "y", "z"]].to_numpy(dtype=float)
    coords_bs2 = bs2[["x", "y", "z"]].to_numpy(dtype=float)

    binding_site_centroid = np.vstack((coords_bs1, coords_bs2)).mean(axis=0)

    # Midpoints from both nearest-neighbor directions: same as the original workflow.
    tree_bs2 = cKDTree(coords_bs2)
    _, nearest_idx2 = tree_bs2.query(coords_bs1, k=1)
    midpoints1 = (coords_bs1 + coords_bs2[nearest_idx2]) / 2.0

    tree_bs1 = cKDTree(coords_bs1)
    _, nearest_idx1 = tree_bs1.query(coords_bs2, k=1)
    midpoints2 = (coords_bs1[nearest_idx1] + coords_bs2) / 2.0

    midpoints = np.vstack((midpoints1, midpoints2))
    centroid, basis = fit_plane(midpoints)

    center_uv = project_point_to_plane(binding_site_centroid, centroid, basis)
    plane_coords1 = project_coords_to_plane(coords_bs1, centroid, basis)
    plane_coords2 = project_coords_to_plane(coords_bs2, centroid, basis)

    circle_radius, _ = build_concentric_rings(
        plane_coords1,
        plane_coords2,
        center_uv,
        n_rings=10,
        min_outer_points=10,
    )
    return circle_radius


# -----------------------------
# Batch execution / CSV output
# -----------------------------

def read_completed_complexes(summary_csv: Path) -> set:
    if not summary_csv.exists() or summary_csv.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(summary_csv, usecols=["complex_name", "radius"])
    except Exception:
        return set()
    valid = df.dropna(subset=["complex_name", "radius"])
    return set(valid["complex_name"].astype(str))


def append_summary_row(summary_csv: Path, complex_name: str, radius: float) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists() or summary_csv.stat().st_size == 0
    with open(summary_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["complex_name", "radius"])
        if write_header:
            writer.writeheader()
        writer.writerow({"complex_name": complex_name, "radius": f"{radius:.10f}"})


def append_failure_row(failure_csv: Path, complex_name: str, error: str) -> None:
    failure_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failure_csv.exists() or failure_csv.stat().st_size == 0
    with open(failure_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["complex_name", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow({"complex_name": complex_name, "error": error})


def run_complex_task(task: dict, threshold: float, dms_bin: str) -> dict:
    start = time.perf_counter()
    complex_name = task["complex_name"]
    try:
        with TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            with zipfile.ZipFile(task["zip_path"], "r") as zf:
                surface1 = extract_zip_member(zf, task["members"][0], tmpdir)
                surface2 = extract_zip_member(zf, task["members"][1], tmpdir)

            radius = compute_radius_for_pair(surface1, surface2, threshold=threshold, dms_bin=dms_bin)

        return {
            "ok": True,
            "complex_name": complex_name,
            "radius": radius,
            "elapsed_seconds": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "ok": False,
            "complex_name": complex_name,
            "error": repr(exc),
            "elapsed_seconds": time.perf_counter() - start,
        }
    finally:
        gc.collect()


def run_batch(args: argparse.Namespace) -> None:
    zip_path = Path(args.zip).expanduser().resolve()
    summary_csv = Path(args.output).expanduser().resolve()
    failure_csv = summary_csv.with_name(summary_csv.stem + "_failures.csv")

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"Input must be a .zip archive: {zip_path}")

    if args.force:
        if summary_csv.exists():
            summary_csv.unlink()
        if failure_csv.exists():
            failure_csv.unlink()

    tasks, skipped = discover_pdb_pairs_from_zip(zip_path)
    completed = read_completed_complexes(summary_csv)
    pending = [task for task in tasks if task["complex_name"] not in completed]

    print(f"Found complete complexes: {len(tasks)}")
    if skipped:
        print(f"Skipped incomplete groups: {len(skipped)} (not exactly 2 PDB files)")
    print(f"Already completed in output: {len(completed)}")
    print(f"Pending complexes: {len(pending)}")
    print(f"Summary CSV: {summary_csv}")

    if not pending:
        print("Nothing to do.")
        return

    done = 0
    failed = 0

    if args.workers == 1:
        for task in pending:
            result = run_complex_task(task, args.threshold, args.dms_bin)
            done += 1
            if result["ok"]:
                append_summary_row(summary_csv, result["complex_name"], result["radius"])
                if args.verbose:
                    print(f"OK {result['complex_name']} radius={result['radius']:.6f} ({result['elapsed_seconds']:.1f}s)")
            else:
                failed += 1
                append_failure_row(failure_csv, result["complex_name"], result["error"])
                print(f"FAILED {result['complex_name']}: {result['error']}", file=sys.stderr)

            if not args.verbose and (done == 1 or done % args.progress_every == 0 or done == len(pending)):
                print(f"Progress: {done}/{len(pending)} completed, failures={failed}")
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1)
        except TypeError:
            executor = ProcessPoolExecutor(max_workers=args.workers)

        futures = {}
        try:
            futures = {
                executor.submit(run_complex_task, task, args.threshold, args.dms_bin): task
                for task in pending
            }
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result["ok"]:
                    append_summary_row(summary_csv, result["complex_name"], result["radius"])
                    if args.verbose:
                        print(f"OK {result['complex_name']} radius={result['radius']:.6f} ({result['elapsed_seconds']:.1f}s)")
                else:
                    failed += 1
                    append_failure_row(failure_csv, result["complex_name"], result["error"])
                    print(f"FAILED {result['complex_name']}: {result['error']}", file=sys.stderr)

                if not args.verbose and (done == 1 or done % args.progress_every == 0 or done == len(pending)):
                    print(f"Progress: {done}/{len(pending)} completed, failures={failed}")
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True)

    print(f"Done. Summary saved to: {summary_csv}")
    if failed:
        print(f"Failures saved to: {failure_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover only complex_name and complementary-plane outer radius from a ZIP of paired PDB complexes."
    )
    parser.add_argument("zip", help="input ZIP archive containing paired PDB files")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output summary CSV path; columns are exactly: complex_name,radius",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=5.0,
        help="binding-site distance threshold in angstroms (default: 5.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes (default: 1; use more for speed if RAM allows)",
    )
    parser.add_argument(
        "--dms-bin",
        default="dms",
        help="DMS executable name/path (default: dms)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing output/failures CSV instead of resuming",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print one line per processed complex",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="progress print frequency when not verbose (default: 25)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")

    run_batch(args)


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows/macOS spawn mode.
    main()
