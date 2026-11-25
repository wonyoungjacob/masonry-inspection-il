"""Full SO(3) box union/intersection volumes for the three-brick task."""
from itertools import combinations
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@lru_cache(maxsize=32)
def triples(n):
    return np.array(list(combinations(range(n), 3)))


def box(center, rotation, half_sizes):
    # Local box coordinates: R.T @ (point - center), bounded by +/- half_sizes.
    normals = np.vstack((rotation.T, -rotation.T))
    offsets = normals @ center + np.tile(half_sizes, 2)
    extent = np.abs(rotation) @ half_sizes
    return normals, offsets, center - extent, center + extent, float(8 * np.prod(half_sizes))


def intersection(boxes):
    if len(boxes) == 1:
        return boxes[0][4]
    if np.any(np.max([b[2] for b in boxes], axis=0) >= np.min([b[3] for b in boxes], axis=0)):
        return 0.0
    normals = np.concatenate([b[0] for b in boxes])
    offsets = np.concatenate([b[1] for b in boxes])
    indices = triples(len(normals))
    matrices = normals[indices]
    good = np.abs(np.linalg.det(matrices)) > 1e-10
    if not good.any():
        return 0.0
    vertices = np.linalg.solve(matrices[good], offsets[indices[good]][..., None])[..., 0]
    vertices = vertices[np.all(vertices @ normals.T <= offsets + 1e-10, axis=1)]
    vertices = np.unique(np.round(vertices, 12), axis=0)
    if len(vertices) < 4:
        return 0.0
    try:
        return float(ConvexHull(vertices).volume)
    except QhullError:
        # Coplanar/collinear intersection has zero volume; full-dimensional errors must surface.
        if np.linalg.matrix_rank(vertices - vertices[0], tol=1e-10) < 3:
            return 0.0
        raise


def union_terms(boxes):
    return [(1 if n % 2 else -1, subset)
            for n in range(1, len(boxes) + 1) for subset in combinations(boxes, n)]


def union_iou(actual, target):
    """Inclusion-exclusion handles physical penetration as well as disjoint boxes."""
    a_terms, t_terms = union_terms(actual), union_terms(target)
    av = sum(sign * intersection(subset) for sign, subset in a_terms)
    tv = sum(sign * intersection(subset) for sign, subset in t_terms)
    iv = sum(sa * st * intersection(aa + tt) for sa, aa in a_terms for st, tt in t_terms)
    denominator = av + tv - iv
    return float(np.clip(iv / denominator, 0, 1)) if denominator > 0 else 0.0
