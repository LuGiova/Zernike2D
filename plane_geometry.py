"""Geometry and sampling helpers for the complementary plane.

Some routines are adapted from the original Zernike2D scripts in this
repository and are kept separate so the main workflow stays focused on orchestration.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
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

    # Adapt min_outer_points for imbalanced binding sites
    # If one binding site is much smaller than the other, reduce the requirement
    total_points1 = len(radii1)
    total_points2 = len(radii2)
    min_total = min(total_points1, total_points2)
    max_total = max(total_points1, total_points2)
    
    # If binding sites are very imbalanced, use a smaller min_outer_points
    if min_total < 50:  # Very small binding site
        adaptive_min_outer_points = max(1, min_total // 10)  # At least 10% of points, min 1
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
            return radius, ring_width, radii1, radii2, ring_ids1, ring_ids2

        radius *= 0.95

    raise ValueError(f'Unable to find a circle radius where the outer ring contains at least {adaptive_min_outer_points} points for both binding sites')


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


def select_ring_pairs_angular_cells(plane_coords1, plane_coords2, coords1, coords2, center_uv, plane_point, plane_normal, basis, radius, ring_ids1, ring_ids2, target_cells, n_rings=10):
    """Sample pairs using radial subrings within each ring.

    Each ring is split into 10 equal-thickness subrings. Every subring is handled
    independently. If a subring contains few points, all of them are matched directly.
    Otherwise, the subring is subdivided into radial cells and the number of cells is
    adapted so that the number of qualifying cells stays close to the requested target
    per subring.
    """
    center_uv = np.asarray(center_uv, dtype=float)
    relative1 = plane_coords1 - center_uv
    relative2 = plane_coords2 - center_uv
    radii1 = np.linalg.norm(relative1, axis=1)
    radii2 = np.linalg.norm(relative2, axis=1)

    n_subrings = 10
    target_per_subring = max(1, int(np.ceil(float(target_cells) / float(n_subrings))))
    lower_bound = max(1, int(np.floor(target_per_subring * 0.9)))
    upper_bound = max(lower_bound, int(np.ceil(target_per_subring * 1.1)))

    records = []
    for ring_id in range(n_rings):
        indices1 = np.flatnonzero(ring_ids1 == ring_id)
        indices2 = np.flatnonzero(ring_ids2 == ring_id)

        if len(indices1) == 0 or len(indices2) == 0:
            continue

        ring_width = float(radius / float(n_rings))
        ring_inner_radius = float(ring_id * ring_width)
        subring_width = float(ring_width / float(n_subrings))

        radii1_ring = radii1[indices1]
        radii2_ring = radii2[indices2]

        for subring_id in range(n_subrings):
            sub_inner = ring_inner_radius + float(subring_id) * subring_width
            sub_outer = sub_inner + subring_width
            if subring_id == n_subrings - 1:
                mask1 = (radii1_ring >= sub_inner) & (radii1_ring <= sub_outer)
                mask2 = (radii2_ring >= sub_inner) & (radii2_ring <= sub_outer)
            else:
                mask1 = (radii1_ring >= sub_inner) & (radii1_ring < sub_outer)
                mask2 = (radii2_ring >= sub_inner) & (radii2_ring < sub_outer)

            sub_indices1 = indices1[mask1]
            sub_indices2 = indices2[mask2]
            if len(sub_indices1) == 0 or len(sub_indices2) == 0:
                continue

            sub_total = len(sub_indices1) + len(sub_indices2)

            if sub_total <= int(np.ceil(target_per_subring * 1.5)):
                distance_matrix = np.linalg.norm(
                    plane_coords1[sub_indices1][:, None, :] - plane_coords2[sub_indices2][None, :, :],
                    axis=2,
                )
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                for row, col in zip(row_ind, col_ind):
                    idx1 = int(sub_indices1[row])
                    idx2 = int(sub_indices2[col])
                    proj1 = plane_coords1[idx1]
                    proj2 = plane_coords2[idx2]
                    intersection_3d = segment_plane_intersection(coords1[idx1], coords2[idx2], plane_point, plane_normal)
                    intersection_uv, intersection_proj = project_point_to_plane(intersection_3d, plane_point, basis)
                    relative_intersection = intersection_uv - center_uv
                    cell_center_uv = (proj1 + proj2) / 2.0

                    records.append({
                        'idx1': idx1,
                        'idx2': idx2,
                        'ring_id': int(ring_id + 1),
                        'ring_fraction': float((ring_id + 1) / n_rings),
                        'circle_radius': float(radius),
                        'ring_width': ring_width,
                        'ring_inner_radius': float(ring_id * ring_width),
                        'ring_outer_radius': float((ring_id + 1) * ring_width),
                        'plane_u1': float(proj1[0]),
                        'plane_v1': float(proj1[1]),
                        'plane_u2': float(proj2[0]),
                        'plane_v2': float(proj2[1]),
                        'plane_u': float(cell_center_uv[0]),
                        'plane_v': float(cell_center_uv[1]),
                        'rep_x': float(intersection_proj[0]),
                        'rep_y': float(intersection_proj[1]),
                        'rep_z': float(intersection_proj[2]),
                        'theta': float(np.arctan2(relative_intersection[1], relative_intersection[0])),
                        'radial_distance': float(np.linalg.norm(relative_intersection)),
                        'ring_radius1': float(np.linalg.norm(proj1 - center_uv)),
                        'ring_radius2': float(np.linalg.norm(proj2 - center_uv)),
                    })
                continue

            preferred_cells = max(2, min(target_per_subring, len(sub_indices1), len(sub_indices2)))
            max_cells = max(2, min(len(sub_indices1), len(sub_indices2), target_per_subring * 3))

            selected = None
            best_candidate = None
            best_score = None

            candidate_offsets = [0]
            for step in range(1, max_cells + 1):
                candidate_offsets.extend([step, -step])

            radii1_sub = radii1[sub_indices1]
            radii2_sub = radii2[sub_indices2]

            for offset in candidate_offsets:
                n_cells = preferred_cells + offset
                if n_cells < 2 or n_cells > max_cells:
                    continue

                bin_edges = np.linspace(sub_inner, sub_outer, n_cells + 1)
                bins1 = np.clip(np.digitize(radii1_sub, bin_edges), 1, n_cells)
                bins2 = np.clip(np.digitize(radii2_sub, bin_edges), 1, n_cells)

                counts1 = np.bincount(bins1, minlength=n_cells + 1)[1:]
                counts2 = np.bincount(bins2, minlength=n_cells + 1)[1:]
                qualifying_mask = (counts1 > 0) & (counts2 > 0)
                qualifying_bins = np.flatnonzero(qualifying_mask) + 1
                qualifying_count = int(len(qualifying_bins))

                score = abs(qualifying_count - target_per_subring)
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = (n_cells, bin_edges, bins1, bins2, qualifying_bins)

                if lower_bound <= qualifying_count <= upper_bound:
                    selected = (n_cells, bin_edges, bins1, bins2, qualifying_bins)
                    break

            if selected is None:
                selected = best_candidate

            n_cells, bin_edges, bins1, bins2, qualifying_bins = selected

            for bin_id in qualifying_bins:
                mask1 = bins1 == bin_id
                mask2 = bins2 == bin_id

                bin_indices1 = sub_indices1[mask1]
                bin_indices2 = sub_indices2[mask2]
                if len(bin_indices1) == 0 or len(bin_indices2) == 0:
                    continue

                bin_center_radius = float(bin_edges[bin_id - 1] + (bin_edges[bin_id] - bin_edges[bin_id - 1]) / 2.0)
                dist1 = np.abs(radii1[bin_indices1] - bin_center_radius)
                dist2 = np.abs(radii2[bin_indices2] - bin_center_radius)
                idx1 = int(bin_indices1[int(np.argmin(dist1))])
                idx2 = int(bin_indices2[int(np.argmin(dist2))])

                proj1 = plane_coords1[idx1]
                proj2 = plane_coords2[idx2]
                intersection_3d = segment_plane_intersection(coords1[idx1], coords2[idx2], plane_point, plane_normal)
                intersection_uv, intersection_proj = project_point_to_plane(intersection_3d, plane_point, basis)
                relative_intersection = intersection_uv - center_uv
                cell_center_uv = (proj1 + proj2) / 2.0

                records.append({
                    'idx1': idx1,
                    'idx2': idx2,
                    'ring_id': int(ring_id + 1),
                    'ring_fraction': float((ring_id + 1) / n_rings),
                    'circle_radius': float(radius),
                    'ring_width': ring_width,
                    'ring_inner_radius': float(ring_id * ring_width),
                    'ring_outer_radius': float((ring_id + 1) * ring_width),
                    'plane_u1': float(proj1[0]),
                    'plane_v1': float(proj1[1]),
                    'plane_u2': float(proj2[0]),
                    'plane_v2': float(proj2[1]),
                    'plane_u': float(cell_center_uv[0]),
                    'plane_v': float(cell_center_uv[1]),
                    'rep_x': float(intersection_proj[0]),
                    'rep_y': float(intersection_proj[1]),
                    'rep_z': float(intersection_proj[2]),
                    'theta': float(np.arctan2(relative_intersection[1], relative_intersection[0])),
                    'radial_distance': float(np.linalg.norm(relative_intersection)),
                    'ring_radius1': float(np.linalg.norm(proj1 - center_uv)),
                    'ring_radius2': float(np.linalg.norm(proj2 - center_uv)),
                })

    return pd.DataFrame.from_records(records)


def select_ring_pairs_kmeans(plane_coords1, plane_coords2, coords1, coords2, center_uv, plane_point, plane_normal, basis, radius, ring_ids1, ring_ids2, n_clusters, n_rings=10):
    """Sample pairs using K-Means clustering of segment-plane intersections.
    
    For each ring:
    - Find nearest neighbor matches from binding site 1 to 2, and vice versa
    - Compute intersection points where connecting segments cross the plane
    - Cluster these intersections using K-Means
        - Select the intersection closest to each cluster centroid as the representative point
            while keeping both binding-site points unique across clusters in the same ring
    """
    center_uv = np.asarray(center_uv, dtype=float)
    
    records = []
    for ring_id in range(n_rings):
        indices1 = np.flatnonzero(ring_ids1 == ring_id)
        indices2 = np.flatnonzero(ring_ids2 == ring_id)
        
        if len(indices1) == 0 or len(indices2) == 0:
            continue
        
        # Collect all intersection points
        intersections_data = []  # List of tuples (idx1, idx2, intersection_uv, intersection_proj)
        
        # Forward matching: bs1 -> bs2
        coords1_ring = plane_coords1[indices1]
        coords2_ring = plane_coords2[indices2]
        
        for i, idx1 in enumerate(indices1):
            # Find nearest neighbor in bs2
            distances = np.linalg.norm(coords1_ring[i:i+1] - coords2_ring, axis=1)
            nearest_j = np.argmin(distances)
            idx2 = indices2[nearest_j]
            
            # Compute intersection
            intersection_3d = segment_plane_intersection(coords1[idx1], coords2[idx2], plane_point, plane_normal)
            intersection_uv, intersection_proj = project_point_to_plane(intersection_3d, plane_point, basis)
            intersections_data.append((idx1, idx2, intersection_uv.copy(), intersection_proj.copy()))
        
        # Reverse matching: bs2 -> bs1
        for i, idx2 in enumerate(indices2):
            # Find nearest neighbor in bs1
            distances = np.linalg.norm(coords2_ring[i:i+1] - coords1_ring, axis=1)
            nearest_j = np.argmin(distances)
            idx1 = indices1[nearest_j]
            
            # Compute intersection
            intersection_3d = segment_plane_intersection(coords1[idx1], coords2[idx2], plane_point, plane_normal)
            intersection_uv, intersection_proj = project_point_to_plane(intersection_3d, plane_point, basis)
            intersections_data.append((idx1, idx2, intersection_uv.copy(), intersection_proj.copy()))
        
        if len(intersections_data) == 0:
            continue
        
        # Extract UV coordinates for clustering
        intersection_uvs = np.array([data[2] for data in intersections_data])
        
        # Determine actual number of clusters (at most number of intersections)
        actual_clusters = min(n_clusters, len(intersections_data))
        used_idx1 = set()
        used_idx2 = set()
        
        if actual_clusters == 1:
            # Single cluster: use the mean intersection
            mean_idx = np.argmin(np.linalg.norm(intersection_uvs - intersection_uvs.mean(axis=0), axis=1))
            idx1, idx2, intersection_uv, intersection_proj = intersections_data[mean_idx]
            proj1 = plane_coords1[idx1]
            proj2 = plane_coords2[idx2]
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
        else:
            # Multiple clusters: use K-Means
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(intersection_uvs)
            
            # For each cluster, find the point closest to the centroid
            for cluster_id in range(actual_clusters):
                cluster_mask = labels == cluster_id
                cluster_intersections = [intersections_data[i] for i in range(len(intersections_data)) if cluster_mask[i]]
                cluster_uvs = intersection_uvs[cluster_mask]
                
                # Find point closest to cluster centroid, but only if both indices are unused
                centroid = kmeans.cluster_centers_[cluster_id]
                distances_to_centroid = np.linalg.norm(cluster_uvs - centroid, axis=1)
                candidate_order = np.argsort(distances_to_centroid)

                selected_candidate = None
                for candidate_idx in candidate_order:
                    candidate = cluster_intersections[candidate_idx]
                    candidate_idx1 = int(candidate[0])
                    candidate_idx2 = int(candidate[1])
                    if candidate_idx1 in used_idx1 or candidate_idx2 in used_idx2:
                        continue
                    selected_candidate = candidate
                    used_idx1.add(candidate_idx1)
                    used_idx2.add(candidate_idx2)
                    break

                if selected_candidate is None:
                    fallback_candidate = cluster_intersections[int(candidate_order[0])]
                    fallback_idx1 = int(fallback_candidate[0])
                    fallback_idx2 = int(fallback_candidate[1])
                    selected_candidate = fallback_candidate
                    used_idx1.add(fallback_idx1)
                    used_idx2.add(fallback_idx2)
                
                idx1, idx2, intersection_uv, intersection_proj = selected_candidate
                proj1 = plane_coords1[idx1]
                proj2 = plane_coords2[idx2]
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
                    'plane_u': float(centroid[0]),
                    'plane_v': float(centroid[1]),
                    'rep_x': float(intersection_proj[0]),
                    'rep_y': float(intersection_proj[1]),
                    'rep_z': float(intersection_proj[2]),
                    'theta': float(np.arctan2(relative_intersection[1], relative_intersection[0])),
                    'radial_distance': float(np.linalg.norm(relative_intersection)),
                    'ring_radius1': float(np.linalg.norm(proj1 - center_uv)),
                    'ring_radius2': float(np.linalg.norm(proj2 - center_uv)),
                })
    
    return pd.DataFrame.from_records(records)