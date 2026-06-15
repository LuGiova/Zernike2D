# Additional Scripts Documentation

**Author:** Giovanni Marzioni (Sapienza University, Matricola 2060629)  
**Course:** Biophysics Laboratory II (Prof. Nucara) with Prof. Milanetti  
**Date:** 2026

This document describes additional scripts developed for interface analysis and surface characterization that extend the core Zernike2D protocol.

---

## Binding Sites

**Script:** `get_binding_site.py`

Identifies and extracts binding site regions between two molecular surfaces. A point is considered part of a binding site if it lies within a specified distance threshold from at least one point on the other surface.

### Arguments

**Required:**
- `-sf1, --surface1` (str): Full path of the surface1 CSV file
- `-sf2, --surface2` (str): Full path of the surface2 CSV file  
- `-o, --output` (str): Destination folder for output files

**Optional:**
- `-t, --threshold` (float): Distance threshold in angstroms (default: 5.0)
- `-p, --plot` (flag): Generate 3D visualization plots
- `-h, --help`: Show help message

### Usage

```bash
python get_binding_site.py -sf1 ./input_files/1a1u_A.csv -sf2 ./input_files/1a1u_C.csv -o ./output_files/ -t 5.0 -p
```

### Output Files

- `<surface1>_bs.csv`: Binding site points from surface 1
- `<surface2>_bs.csv`: Binding site points from surface 2
- Optional: 3D visualization plots (if `-p` flag used)

### Description

The script computes pairwise distances between points on two surfaces and identifies those within the threshold. The output CSV files contain columns: `res, x, y, z, nx, ny, nz` (residue, coordinates, and normal vectors).

---

## Flatness

**Script:** `get_flatness.py`

Computes the flatness metric of a molecular surface using Principal Component Analysis (PCA). Flatness characterizes the surface topology by measuring variation in the z-direction relative to the x-y plane.

### Arguments

**Required:**
- `-s, --surface` (str): CSV file name (e.g., `1a1u_A_bs.csv`)
- `-i, --input` (str): Input path folder with surface file

**Optional:**
- `-h, --help`: Show help message

### Usage

```bash
python get_flatness.py -s 1a1u_A_bs.csv -i ./output_files/
```

### Output

Console output of:
- PC1, PC2, PC3 (eigenvalues from PCA decomposition)
- Flatness coefficient

### Description

The script performs PCA on surface coordinates to determine principal axes. The flatness metric is computed as the ratio of the smallest to largest eigenvalue, providing a measure of how flat the surface is relative to its extent.

---

## Complementary Plane

**Script:** `get_complementary_plane.py`

Computes a complementary plane starting from two **binding-site CSV files** already produced upstream by `get_binding_site.py`. This script is therefore not responsible for detecting the interface: it assumes the user has already extracted the relevant binding-site points and wants to build a plane from that reduced representation.

The full workflow is:

1. Read the two binding-site CSV files.
2. Sample the first binding-site surface uniformly at the requested stride (`--sample-every`).
3. For each sampled point, find the nearest point on the other binding-site surface. Repeat the sampling in the opposite direction (sample the second binding-site surface and find nearest points on the first). If inverse matching produces additional unique pairs, include them as well.
4. Build the midpoint for each matched pair (including the extra inverse pairs).
5. Fit a PCA plane to the midpoint cloud.
6. Use the pair direction as the Zernike axis for each sampled point, or optionally the local surface normal when `--use-surface-normals` is enabled.
7. Project the matched pairs and the plane coordinates for downstream distance analysis.

Because the input is already restricted to binding-site points, this script works on a preselected interface patch. Its defining feature is the way it samples the binding site and how it orients Zernike computation: the local interaction direction comes from the matched pair itself, unless the user asks to use the surface normals instead.

### Arguments

**Required:**
- `-sf1, --surface1` (str): Full path of the surface1 CSV file
- `-sf2, --surface2` (str): Full path of the surface2 CSV file
- `-o, --output` (str): Destination folder for output files

**Optional:**
- `-s, --sample-every` (int): Sample every Nth point (default: 1)
- `--use-surface-normals` (flag): Use surface normals in calculations
- `--output-name` (str): Custom output file name (without extension)
- `-p, --plot` (flag): Generate plots
- `-h, --help`: Show help message

### Usage

```bash
python get_complementary_plane.py -sf1 ./output_files/1a1u_A_bs.csv -sf2 ./output_files/1a1u_C_bs.csv -o ./output_files/ -p
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

Computes a complementary plane starting from the **full molecular surface CSV files**. This script handles the binding-site extraction internally, then builds a circular sampling domain centered on the projected binding-site centroid and computes Zernike descriptors using the plane normal as the axis.

The full workflow is:

1. Read the two full surface CSV files.
2. Convert the Cartesian coordinates to arrays and extract binding sites using the distance threshold.
3. Keep only the points classified as binding-site points on each surface.
4. Match the two binding-site clouds through nearest-neighbor search between the two binding-site sets in both directions.
5. Compute pairwise midpoints and fit a PCA plane to the midpoint cloud. Note: midpoints are collected from both matching directions — points in surface1 matched to their nearest on surface2 and points in surface2 matched to their nearest on surface1 — and the combined midpoint cloud is used for plane fitting.
6. Project the binding-site centroid onto the fitted plane.
7. Build a circle centered on that projected centroid and divide it into 10 concentric rings of equal thickness.
8. Shrink the circle radius until the outer ring contains at least 10 points for each binding site.
9. For each ring, select up to `-n` points from binding site 1 and match them one-to-one to distinct points from binding site 2 in the same ring.
10. For each pair, compute the intersection between the segment connecting the two points and the fitted plane, then export the resulting representative point, distances, and ring metadata.

Compared to `get_complementary_plane.py`, this script differs mainly in how the interface is sampled and how Zernike is oriented. Here the plane is not sampled as a rectangle anymore: the domain is a polar circle centered on the projected binding-site centroid, split into 10 rings, and the plane normal is used as the Zernike axis. This makes the resulting table more directly tied to the fitted complementary plane geometry.

The two scripts are not meant to establish a "better" or "worse" method in general. They are two different ways of building a complementary plane with different input assumptions:

- `get_complementary_plane.py` samples the binding site uniformly and computes Zernike with the pair direction, or optionally the surface normal.
- `get_complementary_plane2.py` samples the fitted plane in polar rings and computes Zernike with the plane normal.

The choice depends on whether you want the analysis anchored on the binding-site geometry itself or on the final fitted plane geometry.

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

- `complementary_plane2.csv` (or custom name): Sampled points on fitting plane with distance metrics
- `complementary_plane2.png`: Polar subplot visualization with 10 concentric rings

### Output CSV Columns

The CSV now includes the original matched points as well as the plane coordinates and ring metadata:

`res1, res2, idx1, idx2, ring_id, ring_fraction, circle_radius, ring_width, ring_inner_radius, ring_outer_radius, x1, y1, z1, x2, y2, z2, mid_x, mid_y, mid_z, center_u, center_v, center_x, center_y, center_z, plane_x, plane_y, plane_z, plane_u, plane_v, theta, radial_distance, physical_distance, zernike_distance`

### Notes

- Uses binding-site matching before fitting the plane
- Forces the selected points from binding site 1 to be paired with distinct nearest neighbors from binding site 2 inside the same ring
- Stores the representative point as the intersection between the segment joining the two 3D points and the fitted plane
- The additional columns make downstream analyses possible on the original 3D point pairs, not only on the plane projection

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
