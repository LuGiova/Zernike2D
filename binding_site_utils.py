"""Binding-site helpers for comparing two molecular surfaces."""

import numpy as np
from scipy.spatial import cKDTree


# Adapted from the binding-site logic used in the original Zernike2D workflow
# by Edoardo Milanetti, Mattia Miotto, Lorenzo Di Rienzo, Michele Monti,
# Giorgio Gosti, and Giancarlo Ruocco.
# Source repository: https://github.com/matmi8/Zernike2D.git
def get_binding_site_mask(coords1, coords2, threshold=5.0):
    tree = cKDTree(coords2)
    min_dist, _ = tree.query(coords1, k=1)
    mask = (min_dist <= threshold).astype(int)
    return mask, min_dist