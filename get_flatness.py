import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from docs import build_cli_flatness


class FlatnessCalculator:
    def __init__(self, surface_file, input_path):
        self.surface_file = surface_file
        self.input_path = input_path
        self.surface = None
        self.eigenvalues = None
        self.flatness = None

    def read_surface(self):
        """
        Read a surface CSV file and keep only the coordinate columns.
        """
        surface_path = Path(self.input_path).joinpath(self.surface_file)
        self.surface = pd.read_csv(surface_path)
        return self.surface

    def compute_pca(self):
        """
        Compute PCA on the x, y, z coordinates using scikit-learn.
        The explained variances are ordered from the largest principal component to the smallest.
        """
        if self.surface is None:
            self.read_surface()

        coords = self.surface[['x', 'y', 'z']].to_numpy(dtype=float)
        if coords.shape[0] < 3:
            raise ValueError('At least 3 points are required to compute PCA')

        pca = PCA(n_components=3)
        pca.fit(coords)
        eigenvalues = pca.explained_variance_

        if np.isclose(eigenvalues[0], 0.0) or np.isclose(eigenvalues[1], 0.0):
            raise ValueError('Flatness is undefined because one or more principal components are zero')

        self.eigenvalues = eigenvalues
        self.flatness = ((eigenvalues[2] / eigenvalues[0]) + (eigenvalues[2] / eigenvalues[1])) / 2.0
        return self.eigenvalues, self.flatness

    def compute(self):
        """
        Read the surface, compute PCA, and print the flatness.
        """
        self.read_surface()
        self.compute_pca()

        print(f'PC1: {self.eigenvalues[0]:.6f}')
        print(f'PC2: {self.eigenvalues[1]:.6f}')
        print(f'PC3: {self.eigenvalues[2]:.6f}')
        print(f'Flatness: {self.flatness:.6f}')


def main():
    args = build_cli_flatness()
    calculator = FlatnessCalculator(args.surface, args.input)
    calculator.compute()


if __name__ == '__main__':
    main()