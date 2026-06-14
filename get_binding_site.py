import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
from docs import build_cli_binding_site
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.seterr(divide='ignore', invalid='ignore')


class BindingSite:
    def __init__(self, surface_file1, surface_file2, output_path, threshold=5.0):
        """
        Initialize BindingSite calculator.
        
        Parameters
        ----------
        surface_file1 : str
            Path to first surface CSV file
        surface_file2 : str
            Path to second surface CSV file
        output_path : str
            Output directory for result files
        threshold : float
            Distance threshold in angstroms (default: 5.0)
        """
        self.surface1 = pd.read_csv(surface_file1)
        self.surface2 = pd.read_csv(surface_file2)
        self.file_name1 = Path(surface_file1).stem
        self.file_name2 = Path(surface_file2).stem
        self.output_path = output_path
        self.threshold = threshold

    @staticmethod
    def get_binding_site_mask(coords1, coords2, threshold=5.0):
        """
        Identify binding site points based on distance threshold.
        
        A point in surface1 is part of the binding site if it has at least
        one neighbor in surface2 within the distance threshold.
        
        Parameters
        ----------
        coords1 : ndarray
            Coordinates of surface1 (N x 3)
        coords2 : ndarray
            Coordinates of surface2 (M x 3)
        threshold : float
            Distance threshold in angstroms
        
        Returns
        -------
        mask : ndarray
            Binary mask (1 if point is in binding site, 0 otherwise)
        """
        dist = cdist(coords1, coords2)
        min_dist = np.min(dist, axis=1)
        mask = (min_dist <= threshold).astype(int)
        return mask, min_dist

    def get_binding_sites(self):
        """
        Compute binding sites for both surfaces and save to CSV.
        
        Returns two CSV files with columns: res, x, y, z, nx, ny, nz
        Only includes points that are within threshold distance of a point
        on the other surface.
        """
        coords1 = self.surface1[['x', 'y', 'z']].to_numpy()
        coords2 = self.surface2[['x', 'y', 'z']].to_numpy()

        print(f'Computing binding sites between {self.file_name1} and {self.file_name2}')
        print(f'Distance threshold: {self.threshold} angstrom')

        # Get binding site masks and minimum distances
        mask1, _ = self.get_binding_site_mask(coords1, coords2, self.threshold)
        mask2, _ = self.get_binding_site_mask(coords2, coords1, self.threshold)

        # Filter to keep only binding site points
        df_bs1 = self.surface1[mask1 == 1][['res', 'x', 'y', 'z', 'nx', 'ny', 'nz']].copy()
        df_bs2 = self.surface2[mask2 == 1][['res', 'x', 'y', 'z', 'nx', 'ny', 'nz']].copy()

        # Save to CSV
        output_file1 = Path(self.output_path).joinpath(f'{self.file_name1}_bs.csv')
        output_file2 = Path(self.output_path).joinpath(f'{self.file_name2}_bs.csv')

        df_bs1.to_csv(output_file1, index=False)
        df_bs2.to_csv(output_file2, index=False)

        print(f'Binding site points in {self.file_name1}: {len(df_bs1)}')
        print(f'Binding site points in {self.file_name2}: {len(df_bs2)}')
        print(f'Output files saved to {self.output_path}')

        return df_bs1, df_bs2

    def plot_binding_sites(self, df_bs1, df_bs2):
        """
        Plot binding site points in 3D space.
        
        Parameters
        ----------
        df_bs1 : DataFrame
            Binding site points of surface1
        df_bs2 : DataFrame
            Binding site points of surface2
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot binding site points
        ax.scatter(df_bs1['x'], df_bs1['y'], df_bs1['z'], 
                  c='red', s=20, alpha=0.6, label=f'{self.file_name1} (n={len(df_bs1)})')
        ax.scatter(df_bs2['x'], df_bs2['y'], df_bs2['z'], 
                  c='blue', s=20, alpha=0.6, label=f'{self.file_name2} (n={len(df_bs2)})')

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Binding Sites (threshold: {self.threshold} Å)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.show()



def main():
    args = build_cli_binding_site()
    bs = BindingSite(args.surface1, args.surface2, args.output, args.threshold)
    df_bs1, df_bs2 = bs.get_binding_sites()
    
    if args.plot:
        bs.plot_binding_sites(df_bs1, df_bs2)


if __name__ == "__main__":
    main()
