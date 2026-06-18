# Additional Scripts Documentation

**Author:** Giovanni Marzioni (Sapienza University, Matricola 2060629)  
**Course:** Biophysics Laboratory II (Prof. Nucara) with Prof. Milanetti  
## Complementary Plane 2

**Script:** `get_complementary_plane2.py`

Purpose
 - Build a complementary plane from two full-surface CSVs, sample it in polar rings, and compare local descriptors between matched points.

Behavior (current implementation)
 - Extracts binding sites from the two input full-surface CSV files using a distance threshold.
 - Matches the two binding-site clouds with nearest-neighbor searches in both directions and collects midpoints.
 - Fits a PCA plane to the combined midpoint cloud and projects the binding-site centroid onto that plane.
 - Constructs a circular domain centered on the projected centroid and splits it into 10 concentric rings of equal thickness.
 - Shrinks the circle radius (if needed) until the outermost ring contains at least 10 points for each binding site.
 - For each ring, selects up to `-n` points from binding site 1 (angularly ordered) and assigns a distinct partner from binding site 2 within the same ring using the Hungarian algorithm (one-to-one matching).
 - For each matched pair computes:
	 - the 3D intersection between the segment joining the two original points and the fitted plane (representative point),
	 - the representative point's projected plane coordinates (UV),
	 - the Euclidean physical distance between the two original 3D points,
	 - the Zernike distance between Zernike descriptors computed with the plane normal as axis.

Console output
 - Primary: two lines printed to stdout containing comma-separated sequences of 10 numbers (rings 1..10):
 	 1) inverse-density-weighted mean physical distances per ring
 	 2) inverse-density-weighted mean Zernike distances per ring
 - Additional informational messages (plane centroid, plane normal, projected centroid, final circle radius) are printed to stdout by the current implementation.

CSV output (`--csv`)
 - The detailed per-pair CSV is written only if the `--csv` flag is provided on the CLI.
 - When written, the CSV begins with commented metadata lines (each starting with `#`) containing:
	 - `center_u,center_v` (plane coordinates of projected centroid)
	 - `center_x,center_y,center_z` (3D coords of projected centroid)
	 - `circle_radius` and `ring_width`
	 - `n_rings` (10)
 - After the header comments the per-pair table is written with a single header row and one row per matched pair.

Per-pair CSV columns (when `--csv` is used)
 - `res1, res2, idx1, idx2, ring_id, ring_fraction, circle_radius, ring_width, ring_inner_radius, ring_outer_radius, plane_u1, plane_v1, plane_u2, plane_v2, plane_u, plane_v, rep_x, rep_y, rep_z, theta, radial_distance, ring_radius1, ring_radius2, x1, y1, z1, x2, y2, z2, physical_distance, zernike_distance`

Files produced
 - When `--csv` is used: `<output_name or surface1_surface2_complementary_plane.csv>` containing the commented metadata header and the per-pair table.
 - When `-p/--plot` is used: a PNG polar plot `<output_name or surface1_surface2_plane_comparison.png>` with two subplots (physical distance and Zernike distance) using the plane-centered polar grid.

CLI (see `docs.py`)
 - Required: `-sf1/--surface1`, `-sf2/--surface2`, `-o/--output`
 - Useful options: `-t/--threshold` (float, default 5.0), `-n/--points` (int, points per ring), `--output-name` (string), `--csv` (flag to enable detailed CSV), `--verbose` (enable extra messages), `-p/--plot` (generate PNG)

Notes and considerations
 - The per-ring summary is computed with inverse-density weights from a 2D KDE on the ring coordinates, so crowded regions contribute less than sparse regions.
 - The script writes the summary CSV by default and will also print other informational messages; use `--csv` to persist the detailed per-pair table with metadata. `--verbose` increases console verbosity.
 - The CSV includes per-pair representative coordinates (`rep_x/rep_y/rep_z`) and projected UV coordinates for both original points and intersections; these are provided to enable downstream analyses without re-projecting data.
 - If you prefer different behavior (for example: suppressing informational prints unless `--verbose` is set, renaming `rep_*` to `intersection_*`, or producing a compact CSV), say which change you want and I will apply it.
```

### Output Files

- `<surface1>_<surface2>_complementary_plane.csv`: Sampled points on plane (or custom name if specified)
- `<surface1>_<surface2>_plane_comparison.png`: Visualization plots

### Output CSV Columns

The CSV now stores both the plane representation and the original matched 3D points:

`res1, res2, idx1, idx2, x1, y1, z1, x2, y2, z2, mid_x, mid_y, mid_z, plane_x, plane_y, plane_z, plane_u, plane_v, physical_distance, zernike_distance`

### Detailed Notes

- `idx1` and `idx2` identify the original rows in the two binding-site CSV files.
- `x1,y1,z1` and `x2,y2,z2` are the original 3D coordinates used to build each midpoint.
- `mid_x,mid_y,mid_z` are the midpoint coordinates used to fit the PCA plane.
- `plane_u,plane_v` are the coordinates in the fitted plane basis.
- `plane_x,plane_y,plane_z` are the 3D coordinates obtained by projecting the midpoint back onto the plane.
- `physical_distance` is the Euclidean distance between the two original 3D points.
- `zernike_distance` is the distance between the corresponding Zernike descriptors.

---

## Complementary Plane 2

**Script:** `get_complementary_plane2.py`

Builds a complementary plane starting from either full-surface CSV files or PDB files. If the inputs are CSVs, they are used directly. If the inputs are PDBs, the script computes the molecular surface in memory using `dms`, then continues from the generated surface data without keeping a user-facing intermediate surface file.

The full workflow is:

1. Detect whether each input is a surface CSV or a PDB file.
2. If an input is a PDB, compute its molecular surface with `dms` and use the resulting surface points in memory.
3. If an input is a PDB, compute its radius of gyration before the binding-site analysis.
4. Extract the binding sites from the two surface sets using the distance threshold.
5. Compute the flatness of each binding site after extraction.
6. Match the two binding-site clouds through nearest-neighbor search between the two binding-site sets in both directions.
7. Compute pairwise midpoints and fit a PCA plane to the midpoint cloud. The combined midpoint cloud from both matching directions is used for plane fitting.
8. Project the binding-site centroid onto the fitted plane.
9. Build a circle centered on that projected centroid and divide it into 10 concentric rings of equal thickness.
10. Shrink the circle radius until the outer ring contains at least 10 points for each binding site.
11. For each ring, select up to `-n` points from binding site 1 and match them one-to-one to distinct points from binding site 2 in the same ring.
12. For each pair, compute the intersection between the segment connecting the two points and the fitted plane, then export the representative point and the distance descriptors.

Compared to `get_complementary_plane.py`, this script samples the fitted plane in polar rings and uses the plane normal as the Zernike axis. The resulting output is split into two files: a detailed per-pair CSV that is written only when `--csv` is used, and a summary CSV that is always written.

### Arguments

**Required:**
- `-sf1, --surface1` (str): Full path of the surface1 CSV file
- `-sf2, --surface2` (str): Full path of the surface2 CSV file
- `-o, --output` (str): Destination folder for output files

**Optional:**
- `-t, --threshold` (float): Distance threshold in angstroms (default: 5.0)
- `-n, --points` (int): Number of points per ring for binding-site 1 (default: 100)
- `--output-name` (str): Custom output file name (without extension)
- `--verbose` (flag): Enable detailed console output
- `-p, --plot` (flag): Generate comparison plots
- `-h, --help`: Show help message

### Usage

```bash
python get_complementary_plane2.py -sf1 ./input_files/1a1u_A.csv -sf2 ./input_files/1a1u_C.csv -o ./output_files/ -t 5.0 -n 1000 -p --output-name complementary_plane2 --verbose
```

### Output Files

- `<output_name or surface1_surface2>_summary.csv`: always written; contains two rows (`weighted` and `normal`) with per-ring summary statistics and global metrics
- `<output_name or surface1_surface2>_complementary_plane.csv` (only with `--csv`): detailed per-pair table with metadata comments at the top
- `<output_name or surface1_surface2>_plane_comparison.png` (only with `-p/--plot`): polar subplot visualization with 10 concentric rings

The summary CSV contains, in order, the per-ring physical and Zernike statistics for rings 1..10, then the global metrics (`gyration_radius`, `flatness`, `roughness`, `roughness_uncertainty`, `scalar_prod`, `scalar_prod_uncertainty`) and a `summary_type` column identifying the weighted or normal row. The weighted row uses inverse-density weights from a 2D KDE on the ring coordinates; the normal row uses the usual arithmetic mean and standard error.

### Output CSV Columns

The detailed CSV written with `--csv` contains exactly these columns:

`idx1, res1, x1, y1, z1, idx2, res2, x2, y2, z2, ring_id, plane_u1, plane_v1, plane_u2, plane_v2, rep_x, rep_y, rep_z, scalar_prod, PC3, physical_distance, zernike_distance`

### Notes

- The summary CSV always includes the weighted and normal rows.
- `gyration_radius` is populated when the input is a PDB; for CSV inputs it is written as unavailable.
- `flatness` is computed after binding-site extraction.
- `PC3` is the mean of the absolute third Zernike coefficient of the two paired points.
- `scalar_prod` is the dot product between the two surface normals at the paired points.
- The representative point is the intersection between the segment joining the two 3D points and the fitted plane.

---

## Complementary Plane Plot

**Script:** `plot_complementary_plane.py`

Generates the 2D complementary-plane scatter plot from a single CSV file (no 3D rendering). The script expects plane coordinates and distance values, then saves one PNG figure with two subplots:

- plane colored by physical distance
- plane colored by Zernike distance

If smoothed columns are present in the CSV, the script uses only those values and marks the figure title as smoothed data. It does not generate a second raw-version figure.

### Arguments

**Required:**
- `-i, --input` (str): Path to complementary-plane CSV file
- `-o, --output` (str): Destination folder for output image

**Optional:**
- `-h, --help`: Show help message

### Usage

```bash
python plot_complementary_plane.py -i ./output_files/complementary_plane2.csv -o ./output_files/
```

### Output Naming

The output filename is automatically derived from the input CSV name:

- input: `complementary_plane2.csv`
- output: `complementary_plane2.png`

### Required CSV Columns

- Always required for plotting coordinates: `plane_u`, `plane_v`
- Distance columns:
	- preferred when available: `smoothed_physical`, `smoothed_zernike`
	- fallback: `physical_distance`, `zernike_distance`

If both smoothed columns are available, raw columns are ignored for plotting.

---

## Gyration Radius

**Script:** `gyration_radius.py`

Calculates the radius of gyration from a PDB structure. This metric characterizes the average distance of atoms from the center of mass and provides a measure of protein compactness.

### Arguments

**Required:**
- `--pdb` (str): Path to PDB file
- `-i, --input` (str): Input directory folder

**Optional:**
- `-h, --help`: Show help message

### Usage

```bash
python gyration_radius.py --pdb 1a1u_A.pdb -i ./input_files/
```

### Output

Console output of:
- Radius of gyration value (in Ångströms)

### Description

The script parses PDB files to extract atomic coordinates, computes the center of mass, and calculates the RMS distance of all atoms from the center. This metric is useful for characterizing protein fold compactness.

---

## Interface Correlation Analysis

**Script:** `analyze_interface_correlations.py`

Analyzes the correlation between physical distances and Zernike distances on a protein-protein interface. The smoothing now operates on the original 3D coordinates of the matched points when those coordinates are available, so it can account for proximity on both protein surfaces rather than only on the fitted plane.

### Arguments

**Required:**
- `-i, --input` (str): Path to input CSV file (e.g., from complementary plane scripts)
- `-o, --output` (str): Path to output figure file

**Optional:**
- `-r, --radius` (float): Smoothing radius in angstroms (default: 6.0)
- `--topo` (flag): Generate the 2x2 kriging topography figure in addition to the standard correlation plot
- `--save-csv` (flag): Save smoothed data to CSV file
- `-h, --help`: Show help message

### Usage

```bash
python analyze_interface_correlations.py -i ./output_files/complementary_plane2.csv -o ./output_files/correlation_plot.png -r 6.0 --save-csv
```

### Output Files

- `<output>.png`: Figure with two subplots (raw and smoothed data)
- `<output_stem>_topograhy.png`: Topographic 2x2 kriging figure produced when `--topo` is used
- `<input_stem>_smoothed.csv` (optional): Smoothed distances if `--save-csv` flag is used (saved next to the input CSV file)

### Output Metrics

Console output of:
- Spearman rho and p-value for raw data
- Spearman rho and p-value for spatially smoothed data

### Description

The script implements spatial smoothing using the original 3D coordinates of the paired interface points. The default strategy is a distance-combined weighting: for each point, nearby candidates on both proteins are collected with a KD-tree, a combined distance is computed from the two 3D distances, and a weight is assigned to each candidate. Points that are close on only one surface are penalized through an explicit distance penalty, so they contribute less than points that are close on both surfaces.

Spearman correlation is preferred over Pearson because it's robust to non-linear relationships.

When `--topo` is enabled, the script still prints the Spearman statistics and saves the standard correlation figure, then generates a 2x2 ordinary-kriging map using `plane_u` and `plane_v` with the following panels: raw physical distance, raw Zernike distance, smoothed physical distance, and smoothed Zernike distance. The topography output uses the same stem as `-o` with `_topograhy.png` appended before the extension.

Transparency details: the kriging variance is converted into per-point transparency separately for each panel. Lower variance values are more opaque, while higher variance values become fully transparent. The transparency curve is non-linear, so intermediate values stay solid longer before fading. Each panel uses its own variance range, and the whole topography figure is rendered on a dark background to preserve contrast.

---

## Workflow Example

The intended workflow is surface selection and quality filtering first, then interface extraction and correlation analysis. For now the scripts operate on one protein complex at a time, but the longer-term goal is a broader screening workflow where gyration radius and flatness are used as distributions to discard complexes that are not suitable for this type of interface analysis.

A complete analysis workflow using the current scripts:

```bash
# 1. Evaluate compactness first
python gyration_radius.py --pdb ./input_files/1a1u_A.pdb -i ./input_files/
python gyration_radius.py --pdb ./input_files/1a1u_C.pdb -i ./input_files/

# 2. Build the molecular surfaces
python get_molecular_surface.py -pdb 1a1u_A.pdb -i ./input_files/ -o ./tutorial/
python get_molecular_surface.py -pdb 1a1u_C.pdb -i ./input_files/ -o ./tutorial/

# 3. Identify binding sites
python get_binding_site.py -sf1 ./input_files/1a1u_A.csv -sf2 ./input_files/1a1u_C.csv -o ./output_files/ -t 5.0 -p

# 4. Evaluate flatness on the two surfaces
python get_flatness.py -s ./input_files/1a1u_A.csv -i ./input_files/
python get_flatness.py -s ./input_files/1a1u_C.csv -i ./input_files/

# 5. Fit a complementary plane using one of the two methods
python get_complementary_plane2.py -sf1 ./input_files/1a1u_A.csv -sf2 ./input_files/1a1u_C.csv -o ./output_files/ -t 5.0 -n 1000 -p --verbose
# or
python get_complementary_plane.py -sf1 ./input_files/1a1u_A.csv -sf2 ./input_files/1a1u_C.csv -o ./output_files/ -p

# 6. Analyze interface correlation
python analyze_interface_correlations.py -i ./output_files/complementary_plane2.csv -o ./output_files/correlation_analysis.png -r 6.0 --save-csv
```

In a future multi-complex analysis, the first and last steps would likely be aggregated into histograms of gyration radius and flatness, which could then be used to filter out complexes that are not globular enough or do not present a sufficiently flat binding site.

---

## CLI Builder (`docs.py`)

All scripts use standardized command-line interface builders defined in `docs.py`. The module provides consistent argument parsing with grouped required and optional arguments:

- **Positional arguments group**: Includes required parameters
- **Optional arguments group**: Includes optional flags and parameters with defaults
- **Help formatting**: Detailed descriptions for each argument with type information

This ensures consistent CLI behavior across all scripts in the Zernike2D package.
