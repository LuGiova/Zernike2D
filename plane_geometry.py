"""Geometry and sampling helpers for the complementary plane.

Some routines are adapted from the original Zernike2D scripts in this
repository and are kept separate so the main workflow stays focused on orchestration.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


def fit_plane(midpoints):
    midpoints = np.asarray(midpoints, dtype=float)
    if len(midpoints) < 3:
        raise ValueError('At least 3 midpoint points are required to fit a plane')

    pca = PCA(n_components=3)
    pca.fit(midpoints)
    centroid = pca.mean_
    basis = pca.components_

    centered = midpoints - centroid
    plane_u = centered @ basis[0]
    plane_v = centered @ basis[1]
    projected = centroid + np.outer(plane_u, basis[0]) + np.outer(plane_v, basis[1])

    return centroid, basis, projected, plane_u, plane_v


def project_surface_to_plane(surface, centroid, basis):
    coords = surface[['x', 'y', 'z']].to_numpy(dtype=float)
    centered = coords - centroid
    plane_u = centered @ basis[0]
    plane_v = centered @ basis[1]
    plane_coords = np.column_stack((plane_u, plane_v))
    projected = centroid + np.outer(plane_u, basis[0]) + np.outer(plane_v, basis[1])
    return coords, plane_coords, projected


def project_point_to_plane(point, centroid, basis):
    point = np.asarray(point, dtype=float)
    centered = point - centroid
    plane_u = float(centered @ basis[0])
    plane_v = float(centered @ basis[1])
    projected = centroid + plane_u * basis[0] + plane_v * basis[1]
    return np.array([plane_u, plane_v], dtype=float), projected


def segment_plane_intersection(point1, point2, plane_point, plane_normal):
    point1 = np.asarray(point1, dtype=float)
    point2 = np.asarray(point2, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)

    direction = point2 - point1
    denominator = float(np.dot(plane_normal, direction))
    if np.isclose(denominator, 0.0):
        return (point1 + point2) / 2.0

    t = float(np.dot(plane_normal, plane_point - point1) / denominator)
    return point1 + t * direction


def build_concentric_rings(plane_coords1, plane_coords2, center_uv, n_rings=10, min_outer_points=10):
    center_uv = np.asarray(center_uv, dtype=float)
    radii1 = np.linalg.norm(plane_coords1 - center_uv, axis=1)
    radii2 = np.linalg.norm(plane_coords2 - center_uv, axis=1)

    max_radius = float(max(np.max(radii1), np.max(radii2)))
    if np.isclose(max_radius, 0.0):
        raise ValueError('Projected binding sites are degenerate in the complementary plane')

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
        if outer_count1 >= min_outer_points and outer_count2 >= min_outer_points:
            return radius, ring_width, radii1, radii2, ring_ids1, ring_ids2

        radius *= 0.95

    raise ValueError('Unable to find a circle radius where the outer ring contains at least 10 points for both binding sites')


def select_ring_pairs(plane_coords1, plane_coords2, coords1, coords2, center_uv, plane_point, plane_normal, basis, radius, ring_ids1, ring_ids2, points_per_ring, n_rings=10):
    center_uv = np.asarray(center_uv, dtype=float)
    relative1 = plane_coords1 - center_uv
    relative2 = plane_coords2 - center_uv
    angle1 = np.arctan2(relative1[:, 1], relative1[:, 0])

    records = []
    for ring_id in range(n_rings):
        indices1 = np.flatnonzero(ring_ids1 == ring_id)
        indices2 = np.flatnonzero(ring_ids2 == ring_id)
        if len(indices1) == 0 or len(indices2) == 0:
            continue

        order1 = indices1[np.argsort(angle1[indices1])]
        target_count = min(points_per_ring, len(order1), len(indices2))
        if target_count < 1:
            continue

        if len(order1) > target_count:
            chosen1 = order1[np.linspace(0, len(order1) - 1, target_count, dtype=int)]
        else:
            chosen1 = order1

        distance_matrix = np.linalg.norm(
            plane_coords1[chosen1][:, None, :] - plane_coords2[indices2][None, :, :],
            axis=2,
        )
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        chosen1 = chosen1[row_ind]
        chosen2 = indices2[col_ind]

        for idx1, idx2 in zip(chosen1, chosen2):
            proj1 = plane_coords1[idx1]
            proj2 = plane_coords2[idx2]
            intersection_3d = segment_plane_intersection(coords1[idx1], coords2[idx2], plane_point, plane_normal)
            intersection_uv, intersection_proj = project_point_to_plane(intersection_3d, plane_point, basis)
            relative_intersection = intersection_uv - center_uv
            records.append({
                'idx1': int(idx1),
                'idx2': int(idx2),
                'ring_id': int(ring_id + 1),
                'ring_fraction': float((ring_id + 1) / n_rings),
                'circle_radius': float(radius),
                'ring_width': float(radius / float(n_rings)),
                'ring_inner_radius': float(ring_id * (radius / float(n_rings))),
                'ring_outer_radius': float((ring_id + 1) * (radius / float(n_rings))),
                'plane_u1': float(proj1[0]),
                'plane_v1': float(proj1[1]),
                'plane_u2': float(proj2[0]),
                'plane_v2': float(proj2[1]),
                'plane_u': float(intersection_uv[0]),
                'plane_v': float(intersection_uv[1]),
                'rep_x': float(intersection_proj[0]),
                'rep_y': float(intersection_proj[1]),
                'rep_z': float(intersection_proj[2]),
                'theta': float(np.arctan2(relative_intersection[1], relative_intersection[0])),
                'radial_distance': float(np.linalg.norm(relative_intersection)),
                'ring_radius1': float(np.linalg.norm(proj1 - center_uv)),
                'ring_radius2': float(np.linalg.norm(proj2 - center_uv)),
            })

    return pd.DataFrame.from_records(records)


def normal_stats(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float('nan'), float('nan')
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return mean_value, 0.0
    uncertainty = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean_value, uncertainty


def weighted_stats(values, plane_coords):
    values = np.asarray(values, dtype=float)
    plane_coords = np.asarray(plane_coords, dtype=float)

    if len(values) == 0:
        return float('nan'), float('nan')
    if len(values) < 3:
        return normal_stats(values)

    try:
        kde = gaussian_kde(plane_coords.T)
        density = kde(plane_coords.T)
    except (np.linalg.LinAlgError, ValueError):
        return normal_stats(values)

    density = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(density)) or np.allclose(density, 0.0):
        return normal_stats(values)

    weights = 1.0 / (density + np.finfo(float).eps)
    weight_sum = float(np.sum(weights))
    if np.isclose(weight_sum, 0.0):
        return normal_stats(values)

    weights = weights / weight_sum
    weighted_mean = float(np.sum(weights * values))
    sum_w2 = float(np.sum(weights ** 2))
    effective_n = (1.0 / sum_w2) if not np.isclose(sum_w2, 0.0) else float(len(values))
    variance = float(np.sum(weights * (values - weighted_mean) ** 2))
    uncertainty = float(np.sqrt(variance / effective_n)) if effective_n > 0 else float('nan')
    return weighted_mean, uncertainty