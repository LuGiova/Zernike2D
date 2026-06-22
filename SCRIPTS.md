# Additional Scripts Documentation

**Author:** Giovanni Marzioni (Sapienza University, Matricola 2060629)
**Course:** Biophysics Laboratory II (Prof. Nucara) with Prof. Milanetti

## Complementary Plane Workflow

**Main script:** `get_complementary_plane.py`

Purpose
- Build a complementary plane from two molecular surfaces.
- Sample matched interface points in concentric rings on the plane.
- Compare physical distance and Zernike descriptor distance with configurable sampling strategies.
- Support both single-complex analysis and high-throughput batch processing (directory or zip).

Scope
- This document describes `get_complementary_plane.py` and all local modules directly imported by it:
	- `binding_site_utils.py`
	- `complementary_plane_cli.py`
	- `plane_geometry.py`
	- `plane_plotting.py`
	- `surface_processing.py`

---

## Script: get_complementary_plane.py

### Core Behavior (Current Implementation)

For each protein pair, the script does the following:

1. Load both inputs with automatic format handling:
	 - CSV: reads the surface table directly.
	 - PDB: runs `dms`, parses generated surface points, computes gyration radius from atoms.
2. Extract the two binding-site subsets using nearest-neighbor distance thresholding (both directions).
3. Build midpoint clouds from symmetric NN matching (`bs1 -> bs2` and `bs2 -> bs1`).
4. Fit a PCA plane on the combined midpoint cloud.
5. Project the binding-site centroid on the fitted plane and define concentric rings.
6. Shrink circle radius iteratively until the outer ring has enough points on both sides.
7. Apply one sampling strategy (`default`, `angular_cells`, or `kmeans`) to generate matched pairs per ring.
8. For matched pairs, compute:
	 - representative point (`rep_x`, `rep_y`, `rep_z`) from segment-plane intersection,
	 - physical distance between 3D points,
	 - Zernike distance using plane-normal aligned local axes (via `zepyros`),
	 - normal scalar product,
	 - PC3-derived roughness metrics.

Note on coordinates:
- `rep_x/rep_y/rep_z` is always the 3D segment-plane intersection, including `angular_cells`.
- In `angular_cells`, `plane_u/plane_v` is stored as cell-center UV (`(proj1 + proj2)/2`), not as UV of the intersection.
9. Write summary CSV (always in single mode), optionally detailed CSV (`--csv`), optionally polar plot (`-p`).

---

## Sampling Strategies (Detailed)

The argument `--sampling-strategy` controls how pairs are selected ring-by-ring. The argument `-n/--points` is interpreted differently depending on the strategy.

### 1) `default` strategy

Function used:
- `plane_geometry.select_ring_pairs`

How it works:
- For each ring, points from binding site 1 are sorted by polar angle around the projected center.
- Up to `n` points are selected from site 1 with angular spacing.
- Selected site-1 points are matched one-to-one with site-2 points in the same ring using Hungarian assignment (`linear_sum_assignment`) on plane distance.

Effect of `-n`:
- Target upper bound of selected pairs per ring.
- Actual count is limited by availability in both ring subsets.

Summary behavior:
- Produces **two summary rows**:
	- `weighted`: inverse-density weighted means/uncertainties (KDE-based)
	- `normal`: arithmetic mean with standard error

Hungarian matching (short explanation):
- It solves a minimum-cost one-to-one assignment on a distance matrix.
- It is used specifically to prevent multiple bs1 points from choosing the same bs2 point.

### 2) `angular_cells` strategy

Function used:
- `plane_geometry.select_ring_pairs_angular_cells`

How it works:
- Each of the 10 rings is further split into 10 radial subrings.
- Inside each subring:
	- if few points are available, direct Hungarian matching is applied,
	- otherwise, adaptive radial bins are created to keep qualifying bins near a target count.
- Each qualifying bin yields one representative pair based on proximity to bin-center radius.

Effect of `-n`:
- Interpreted as **target cells per ring**.
- Internally converted to target per subring (`ceil(n / 10)`) and adapted by local occupancy.

Summary behavior:
- Produces **one summary row**:
	- `normal` only (no weighted row)

### 3) `kmeans` strategy

Function used:
- `plane_geometry.select_ring_pairs_kmeans`

How it works:
- For each ring, generate candidate pair intersections using nearest-neighbor matching in both directions.
- Cluster all ring intersections in UV space with K-Means.
- For each cluster, pick the candidate closest to cluster centroid.
- Try to keep indices unique (`idx1`, `idx2`) across selected representatives in the same ring.

Effect of `-n`:
- Interpreted as **number of clusters per ring**.
- Actual clusters are capped by available candidate intersections.

Summary behavior:
- Produces **one summary row**:
	- `normal` only (no weighted row)

### Ring Construction Notes (applies to all strategies)

- Rings are always `n_rings = 10`.
- Radius starts from max projected distance and shrinks by factor `0.95` until outer ring occupancy condition is met.
- Outer-ring minimum is adaptive for very small/imbalanced binding sites.

---

## Console Output: What Is Printed and When

### Single-complex mode

Default (`--verbose` not set):
- Usually no informational prints.
- Errors are raised if a step fails (for example empty binding site, no sampled pairs, zepyros missing).

With `--verbose`:
- Progress/log messages including:
	- input loading and threshold info,
	- binding-site counts,
	- midpoint counts,
	- selected strategy,
	- selected pairs per ring,
	- matched/unique points,
	- final plane centroid/normal,
	- projected center and circle radius,
	- saved-file paths.
- Zernike descriptor loops show `tqdm` progress bars.

### Batch mode (`--batch-dir` or `--batch-zip`)

Default (`--verbose` not set):
- Worker stdout/stderr is suppressed.
- Main process shows a global `tqdm` progress bar (`Complexes ...`).

With `--verbose`:
- Progress bar is disabled.
- Detailed textual status is printed:
	- total complete complexes found,
	- skipped incomplete groups,
	- already completed complexes from resume summary,
	- pending complex count,
	- per-complex `OK` or `FAILED` with elapsed seconds,
	- final summary/failure paths.

---

## CLI and Execution Modes

CLI builder:
- `complementary_plane_cli.py` (`build_cli_complementary_plane2`)

### Single-complex mode

Required:
- `-sf1/--surface1`: first input file (`.csv` or `.pdb`)
- `-sf2/--surface2`: second input file (`.csv` or `.pdb`)
- `-o/--output`: output directory

Optional:
- `-t/--threshold` (float, default `5.0`)
- `-n/--points` (int, default `100`)
- `--sampling-strategy` (`default`, `angular_cells`, `kmeans`)
- `--output-name` (custom stem)
- `--csv` (write detailed per-pair CSV)
- `-p/--plot` (write polar plot PNG)
- `--verbose`

Example:

```bash
python get_complementary_plane.py \
	-sf1 ./input_files/1a1u_A.csv \
	-sf2 ./input_files/1a1u_C.csv \
	-o ./output_files/default \
	-t 5.0 -n 100 --sampling-strategy default --csv -p --verbose
```

### Batch mode with directory

Enable batch by passing exactly one of:
- `--batch-dir <folder>`
- `--batch-zip <archive.zip>`

In batch mode:
- do not pass `--surface1/--surface2`
- do not pass `--output-name`
- each valid complex must resolve to exactly two `.pdb` files grouped by stem convention

Additional batch options:
- `--workers` (default `1`)
- `--batch-summary` (default `<output>/batch_summary.csv`)
- `--force` (ignore previous summary and recompute all)

---

## Batch Mode with ZIP (Focused)

### Input discovery logic

- The zip is scanned recursively.
- Only `.pdb` members are considered.
- Files are grouped by complex stem using `COMPLEX_CHAIN` parsing:
	- examples: `MYSET/1abc_A.pdb`, `1abc_B.pdb` are grouped under `1abc`.
- Groups with anything other than exactly 2 files are skipped.

### Execution model

- Each pending complex is executed independently.
- For zip input, each member is extracted into a temporary directory per task.
- Outputs per complex are written inside `<output>/details/`.
- Per-complex summary rows are appended to one global summary file.

### Resume and fault tolerance

- Existing summary is parsed to detect already completed complexes.
- Expected rows per complex depend on strategy:
	- `default`: 2 rows (`weighted`, `normal`)
	- others: 1 row (`normal`)
- If a previous summary looks partial/corrupted, backup files are created and active summary is cleaned.
- Failed complexes are appended to `<output>/batch_failures.csv`.

### Parallel processing

- `--workers 1`: serial processing.
- `--workers > 1`: `ProcessPoolExecutor`, optionally with `max_tasks_per_child=1` when supported for cleaner RAM recycling.

### Example (zip)

```bash
python get_complementary_plane.py \
	--batch-zip ./input_files/complexes.zip \
	-o ./output_files/batch_kmeans \
	--sampling-strategy kmeans -n 80 \
	--workers 4 --csv -p
```

---

## Output Files

### Single-complex mode

Always written:
- `<stem>_summary.csv`

Optional:
- detailed CSV (only with `--csv`):
	- if `--output-name` is set: `<output-name>.csv`
	- else: `<surface1_stem>_<surface2_stem>_complementary_plane.csv`
- plot PNG (only with `-p`):
	- if `--output-name` is set: `<output-name>.png`
	- else: `<surface1_stem>_<surface2_stem>_plane_comparison.png`

### Batch mode

Always written:
- global summary: `<output>/batch_summary.csv` (or custom path with `--batch-summary`)

On failures:
- `<output>/batch_failures.csv`

Per-complex detailed outputs (`--csv` and/or `-p`):
- saved in `<output>/details/` with complex-based stems.

---

## CSV Schemas

### Detailed per-pair CSV (`--csv`)

Metadata header lines (prefixed `#`):
- center in UV and XYZ
- circle radius
- ring width
- number of rings

Columns:
- `idx1, res1, x1, y1, z1, idx2, res2, x2, y2, z2, ring_id, plane_u1, plane_v1, PC3_1, plane_u2, plane_v2, PC3_2, rep_x, rep_y, rep_z, scalar_prod, physical_distance, zernike_distance`

### Summary CSV

Per ring (`1..10`) for both metrics:
- `physical_ring{r}_mean`
- `physical_ring{r}_uncertainty`
- `zernike_ring{r}_mean`
- `zernike_ring{r}_uncertainty`

Global columns:
- `gyration_radius`
- `gyration_radius_note` (`csv_input`, `pdb_mean`, `mixed_input`)
- `flatness`
- `PC3`
- `roughness`
- `scalar_prod`
- `scalar_prod_uncertainty`
- `summary_type` (`weighted` or `normal`)

In batch summary, extra leading columns are added:
- `complex_name`
- `protein1_file`
- `protein2_file`
- `elapsed_seconds`

---

## Imported Modules: Responsibilities

## 1) binding_site_utils.py

Main function:
- `get_binding_site_mask(coords1, coords2, threshold=5.0)`

Role:
- Builds a KD-tree on `coords2`.
- Computes nearest-neighbor distance for each point in `coords1`.
- Returns:
	- binary mask (`1` if min distance <= threshold)
	- min-distance array

Use in workflow:
- Called twice (both directions) to define interface points on each surface.

## 2) complementary_plane_cli.py

Main function:
- `build_cli_complementary_plane2()`

Role:
- Defines argument groups and validates mode consistency.
- Enforces mutual exclusivity between `--batch-dir` and `--batch-zip`.
- Enforces single-mode vs batch-mode constraints.

## 3) plane_geometry.py

Core geometry:
- `fit_plane`
- `project_surface_to_plane`
- `project_point_to_plane`
- `segment_plane_intersection`
- `build_concentric_rings`

Sampling and stats:
- `select_ring_pairs` (`default`)
- `select_ring_pairs_angular_cells` (`angular_cells`)
- `select_ring_pairs_kmeans` (`kmeans`)
- `normal_stats`
- `weighted_stats`

## 4) plane_plotting.py

Main function:
- `plot_plane_subplots`

Role:
- Produces one PNG with two polar subplots:
	- colored by physical distance
	- colored by Zernike distance
- Uses ring radius for radial grid and projected center for angle/radius conversion.

## 5) surface_processing.py

Input and preprocessing:
- `load_surface_input`:
	- CSV path -> direct table load
	- PDB path -> run `dms`, parse surface, compute gyration radius

Helpers:
- `read_pdb_atoms`
- `parse_dms_surface`
- `calculate_gyration_radius`
- `calculate_flatness`

Notes:
- PDB handling requires external command-line tool `dms` available in environment.

---

## Practical Notes and Constraints

- `zepyros` is required for Zernike descriptors at runtime.
- If `zepyros` is missing, CLI discovery/help still works, but descriptor computation fails when executed.
- Detailed CSV and plot are opt-in; summary is always produced (single mode) or appended globally (batch mode).
- In default strategy, weighted statistics use inverse 2D KDE density so sparse regions contribute more than dense clusters.
- For very small binding sites, outer-ring occupancy requirement is automatically reduced to avoid premature failure.

---

## Quick Usage Patterns

Single complex, baseline weighted+normal summary:

```bash
python get_complementary_plane.py \
	-sf1 ./input_files/1a1u_A.csv \
	-sf2 ./input_files/1a1u_C.csv \
	-o ./output_files/default \
	--sampling-strategy default -n 100 --csv -p
```

Single complex, radial-cell sampling:

```bash
python get_complementary_plane.py \
	-sf1 ./input_files/1a1u_A.csv \
	-sf2 ./input_files/1a1u_C.csv \
	-o ./output_files/angular \
	--sampling-strategy angular_cells -n 120 --verbose
```

Batch from zip, resumable run:

```bash
python get_complementary_plane.py \
	--batch-zip ./input_files/ppi_batch.zip \
	-o ./output_files/batch_default \
	--sampling-strategy default -n 100 \
	--workers 4 --batch-summary ./output_files/batch_default/summary.csv
```
