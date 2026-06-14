import pandas as pd
import numpy as np
from pathlib import Path
from docs import build_cli_gyration_radius


class GyractionRadius:
    def __init__(self, pdb, input_path):
        self.pdb = pdb
        self.input_path = input_path
        self.coordinates = None
        self.rg = None

    def read_pdb(self):
        """
        Reads a PDB file and extracts atomic coordinates
        Returns a DataFrame with columns: ['atom', 'x', 'y', 'z']
        """
        pdb_path = Path(self.input_path).joinpath(self.pdb)
        
        atoms_data = []
        with open(pdb_path, 'r') as file:
            for line in file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        # PDB format: fixed column positions
                        atom_name = line[12:16].strip()
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atoms_data.append({
                            'atom': atom_name,
                            'x': x,
                            'y': y,
                            'z': z
                        })
                    except (ValueError, IndexError):
                        continue
        
        self.coordinates = pd.DataFrame(atoms_data)
        return self.coordinates

    def calculate_gyration_radius(self):
        """
        Calculate the radius of gyration of the protein
        Rg = sqrt(sum((ri - Rcm)^2) / N+1)
        where ri are atomic coordinates, Rcm is center of mass, N+1 is number of atoms (N bonds)
        """
        if self.coordinates is None:
            self.read_pdb()
        
        # Calculate center of mass
        coords = self.coordinates[['x', 'y', 'z']].values
        center_of_mass = np.mean(coords, axis=0)
        
        # Calculate distances from center of mass
        distances = coords - center_of_mass
        squared_distances = np.sum(distances ** 2, axis=1)
        
        # Calculate radius of gyration
        self.rg = np.sqrt(np.mean(squared_distances))
        
        return self.rg

    def compute(self):
        """
        Main method that orchestrates the calculation and prints the result
        """
        self.read_pdb()
        self.calculate_gyration_radius()
        print(f"Radius of gyration: {self.rg:.4f}")


def main():
    args = build_cli_gyration_radius()
    gr = GyractionRadius(args.pdb, args.input)
    gr.compute()


if __name__ == "__main__":
    main()
