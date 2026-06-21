"""Surface loading and surface-shape helpers.

Portions of these helpers are adapted from the original scripts in this
repository, especially get_molecular_surface.py.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# Adapted from get_molecular_surface.py in the original Zernike2D workflow
# by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
# Giorgio Gosti, and Giancarlo Ruocco.
# Source repository: https://github.com/matmi8/Zernike2D.git
def range_char(start, stop):
    return (chr(n) for n in range(ord(start), ord(stop) + 1))


# Adapted from get_molecular_surface.py in the original Zernike2D workflow
# by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
# Giorgio Gosti, and Giancarlo Ruocco.
# Source repository: https://github.com/matmi8/Zernike2D.git
def read_pdb_atoms(pdb_path):
    atoms_data = []
    with open(pdb_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_name = line[12:16].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atoms_data.append({'atom': atom_name, 'x': x, 'y': y, 'z': z})
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame(atoms_data)


# Adapted from get_molecular_surface.py in the original Zernike2D workflow
# by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
# Giorgio Gosti, and Giancarlo Ruocco.
# Source repository: https://github.com/matmi8/Zernike2D.git
def parse_dms_surface(dms_path):
    colnames = [character for character in range_char('A', 'K')]
    
    # Read DMS with flexible column handling for robustness
    # Some DMS files have variable formatting that creates extra fields
    dms_surf = pd.read_csv(dms_path, names=colnames, header=None, delimiter=r'\s+', 
                           usecols=list(range(len(colnames))))

    # Clean only residue identifier columns so numeric columns keep their decimals
    dms_surf[['A', 'B', 'C']] = dms_surf[['A', 'B', 'C']].astype(str).replace(r'[^\w\s]|_', '', regex=True)

    df = (dms_surf
          .assign(res=np.where(
              dms_surf['B'].astype(str).str[-1].str.isnumeric(),
              dms_surf['A'].astype(str) + '_' +
              dms_surf['B'].astype(str) + '_' +
              dms_surf['C'].astype(str),
              dms_surf['A'].astype(str) + '_' +
              (dms_surf['B'].astype(str).str
               .extract(r'(\d+\.?\d*)([A-Za-z]*)', expand=True)
               .agg('_'.join, axis=1)) + '_' +
              dms_surf['C'])
          )
          .query('G.str[0] == "S"')
          .filter(items=['res', 'D', 'E', 'F', 'I', 'J', 'K'])
          .set_axis(['res', 'x', 'y', 'z', 'nx', 'ny', 'nz'], axis=1)
          .reset_index(drop=True))
    return df


# Adapted from get_molecular_surface.py in the original Zernike2D workflow
# by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
# Giorgio Gosti, and Giancarlo Ruocco.
# Source repository: https://github.com/matmi8/Zernike2D.git
def calculate_gyration_radius(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return float('nan')
    center_of_mass = np.mean(coords, axis=0)
    squared_distances = np.sum((coords - center_of_mass) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distances)))


def calculate_flatness(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 3:
        return np.nan
    pca = PCA(n_components=3)
    pca.fit(coords)
    eigenvalues = pca.explained_variance_
    if np.isclose(eigenvalues[0], 0.0) or np.isclose(eigenvalues[1], 0.0):
        return np.nan
    return float(((eigenvalues[2] / eigenvalues[0]) + (eigenvalues[2] / eigenvalues[1])) / 2.0)


# Adapted from get_molecular_surface.py in the original Zernike2D workflow
# by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
# Giorgio Gosti, and Giancarlo Ruocco.
# Source repository: https://github.com/matmi8/Zernike2D.git
def load_surface_input(input_path):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(input_path).reset_index(drop=True), {'kind': 'csv', 'gyration_radius': np.nan, 'gyration_radius_note': 'csv_input'}
    if suffix == '.pdb':
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dms_output = tmpdir / f'{input_path.stem}.dms'
            run_dms = ['dms', str(input_path), '-n', '-o', str(dms_output)]
            subprocess.run(run_dms, check=True)

            with open(dms_output, 'r') as file:
                file_lines = [f'{x[:14]} {x[14:]}' for x in file.readlines()]
            with open(dms_output, 'w') as file:
                file.writelines(file_lines)

            surface = parse_dms_surface(dms_output)
        atoms = read_pdb_atoms(input_path)
        gyration_radius = calculate_gyration_radius(atoms[['x', 'y', 'z']].to_numpy(dtype=float))
        return surface, {'kind': 'pdb', 'gyration_radius': gyration_radius, 'gyration_radius_note': 'pdb_input'}

    raise ValueError(f'Unsupported input file type: {input_path.suffix}')