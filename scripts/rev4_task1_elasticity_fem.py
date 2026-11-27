"""Revision v4 Task 1 (P0): Unstructured Triangular FEM 2D Plane-Strain Linear Elasticity Benchmark.

Domain: Unit square plate with central circular hole Omega = (0, 1)^2 \\ B_{0.20}(0.5, 0.5).
BCs: Left edge x = 0 clamped Dirichlet u = 0; right edge x = 1 spatially varying random Neumann
traction t(y) = (t_x(y), t_y(y)); top/bottom edges and circular hole traction-free.
Heterogeneous Young's modulus E(x, y) = exp(g(x, y)) and spatially constant random Poisson ratio nu_P.

Formal seeds locked to [42, 137, 314, 271, 811]; strict Train / Val / Cal / Test isolation.
Includes:
  - True locally-refined Delaunay + boundary-locked Laplacian/CVT unstructured P1 meshes (T_c, T_f,
    plus 5 multiscale meshes up to ~45k DOFs) with strict quality & hanging-node unit checks;
  - Exact P1 plane-strain FEM assembly + double-precision Patch Test (< 1e-12) + reference spsolve;
  - Mesh-independent Vector DeepONet (1 frozen model + 5-member ensemble per seed, FP32 no TF32);
  - Numba-accelerated 2-DOF nodal block Smoothed-Aggregation Galerkin AMG (Vector SA-AMG-V2)
    with Power-Iteration global energy contraction verification (||I - B K||_K < 1) & O(M) scaling;
  - Mass-weighted cKDTree physical envelope + derived von Mises stress simultaneous bands;
  - Fair comparison across 9 baselines (OpRes-CP, Exact oracle, Unsigned AMG, Raw |R| PI-CP,
    Babuska-Rheinboldt / Verfurth residual indicator + CP, Zienkiewicz-Zhu ZZ + CP,
    Jha corrector + Uniform/OpRes-CP, 5-Member Ensemble, Uniform-CP).

Outputs written to results/phase3_revision_v4/task1_elasticity.json and elasticity_example_s42.npz.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from numba import njit
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import torch
from torch import nn
from torch.nn import functional as F

from darcy_day15 import value_at_coverage
from darcy_fno import atomic_checkpoint
from darcy_phase0_eval import quantile
from phase3_common import finite_json
from phase3_summarize import SEEDS, stat
from rev2_task1_fidelity import compute_scale_metrics, compute_signed_recovery


OUT = Path("results/phase3_revision_v4")
CACHE = OUT / "cache" / "elasticity"

HOLE_CX = 0.5
HOLE_CY = 0.5
HOLE_R = 0.20

ELAST_CFG = {
    "seeds": list(SEEDS),
    "counts": {
        "train": 1000,
        "val": 200,
        "cal": 400,
        "id": 500,
        "paired": 500,
        "ood": 500,
    },
    "streams": {
        "train": 9601,
        "val": 9602,
        "cal": 9603,
        "id": 9604,
        "paired": 9605,
        "ood": 9606,
    },
    "sigma_log_e_id": [0.25, 0.55],
    "sigma_log_e_ood": [0.55, 0.85],
    "nu_id": [0.20, 0.34],
    "nu_ood": [0.34, 0.43],
    "moduli_modes": 12,
    "moduli_length": 0.15,
    "alpha": 0.10,
    "relative_floor": 0.10,
    "abs_floor": 1e-10,
    "lambda": 0.85,
    "ensemble_offsets": [0, 10000, 20000, 30000, 40000],
    "model": {
        "epochs": 240,
        "batch_size": 64,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "width_branch": 256,
        "width_trunk": 192,
        "latent_dim": 96,
    },
}


def _seed_stats(values):
    _, avg, sd, lo, hi = stat(values)
    return {
        "mean": float(avg),
        "sd": float(sd),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "per_seed": {str(s): float(v) for s, v in zip(SEEDS, values)},
    }


# ---------------------------------------------------------------------------
# 1. Unstructured Triangular P1 Mesh Generation & Quality Verification
# ---------------------------------------------------------------------------

def _target_spacing(x, y, h_hole, h_outer):
    r = np.hypot(x - HOLE_CX, y - HOLE_CY)
    t = np.clip((r - HOLE_R) / 0.25, 0.0, 1.0)
    return h_hole + (h_outer - h_hole) * t


def _tri_angles_and_areas(pts, tris):
    p0 = pts[tris[:, 0]]
    p1 = pts[tris[:, 1]]
    p2 = pts[tris[:, 2]]
    v01 = p1 - p0
    v12 = p2 - p1
    v20 = p0 - p2
    signed_area = 0.5 * (v01[:, 0] * (-v20[:, 1]) - v01[:, 1] * (-v20[:, 0]))
    l0 = np.hypot(v12[:, 0], v12[:, 1])
    l1 = np.hypot(v20[:, 0], v20[:, 1])
    l2 = np.hypot(v01[:, 0], v01[:, 1])
    c0 = np.clip((l1 * l1 + l2 * l2 - l0 * l0) / np.maximum(2.0 * l1 * l2, 1e-30), -1.0, 1.0)
    c1 = np.clip((l0 * l0 + l2 * l2 - l1 * l1) / np.maximum(2.0 * l0 * l2, 1e-30), -1.0, 1.0)
    c2 = np.clip((l0 * l0 + l1 * l1 - l2 * l2) / np.maximum(2.0 * l0 * l1, 1e-30), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(np.stack((c0, c1, c2), axis=1)))
    return signed_area, angles_deg


def _filter_plate_tris(pts, simplices):
    cent = pts[simplices].mean(axis=1)
    rc = np.hypot(cent[:, 0] - HOLE_CX, cent[:, 1] - HOLE_CY)
    keep = rc > HOLE_R + 1e-9
    tris = simplices[keep].copy()
    area, _ = _tri_angles_and_areas(pts, tris)
    flip = area < 0.0
    if np.any(flip):
        tris[flip, 1], tris[flip, 2] = tris[flip, 2], tris[flip, 1]
    return tris


def build_perforated_plate_mesh(h_hole, h_outer, n_edge, n_hole, smooth_iters=24, verify=True):
    """Build locally refined unstructured P1 mesh on (0,1)^2 \\ B_0.20(0.5, 0.5)."""
    # Outer square boundary (counter-clockwise without duplicate corners)
    s = np.linspace(0.0, 1.0, n_edge + 1)[:-1]
    b_bot = np.column_stack((s, np.zeros_like(s)))
    b_right = np.column_stack((np.ones_like(s), s))
    b_top = np.column_stack((1.0 - s, np.ones_like(s)))
    b_left = np.column_stack((np.zeros_like(s), 1.0 - s))
    outer_pts = np.vstack((b_bot, b_right, b_top, b_left))

    # Circular hole boundary (clockwise or CCW)
    th = np.linspace(0.0, 2.0 * math.pi, n_hole, endpoint=False)
    hole_pts = np.column_stack((HOLE_CX + HOLE_R * np.cos(th), HOLE_CY + HOLE_R * np.sin(th)))
    n_bnd = len(outer_pts) + len(hole_pts)

    # Smoothly graded concentric rings around the circular hole
    rings = []
    r_cur = HOLE_R + 0.88 * h_hole
    phase = 0.5
    while r_cur < 0.305:
        hr = float(_target_spacing(HOLE_CX + r_cur, HOLE_CY, h_hole, h_outer))
        n_ring = max(12, int(round(2.0 * math.pi * r_cur / hr)))
        ang = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False) + phase * math.pi / n_ring
        rings.append(np.column_stack((HOLE_CX + r_cur * np.cos(ang), HOLE_CY + r_cur * np.sin(ang))))
        r_cur += 0.88 * hr
        phase = 1.0 - phase
    ring_pts = np.vstack(rings) if rings else np.zeros((0, 2))
    r_ring_max = float(np.hypot(ring_pts[:, 0] - HOLE_CX, ring_pts[:, 1] - HOLE_CY).max()) if len(ring_pts) else HOLE_R

    # Outer hex-packed interior points
    dy = h_outer * math.sqrt(3.0) * 0.5
    ys = np.arange(0.55 * dy, 1.0 - 0.45 * dy, dy)
    cand = []
    for row_idx, y in enumerate(ys):
        x0 = 0.55 * h_outer + (0.5 * h_outer if (row_idx & 1) else 0.0)
        xs = np.arange(x0, 1.0 - 0.45 * h_outer, h_outer)
        if len(xs):
            cand.append(np.column_stack((xs, np.full_like(xs, y))))
    hex_pts = np.vstack(cand)
    r_hex = np.hypot(hex_pts[:, 0] - HOLE_CX, hex_pts[:, 1] - HOLE_CY)
    d_outer = np.minimum.reduce((hex_pts[:, 0], 1.0 - hex_pts[:, 0], hex_pts[:, 1], 1.0 - hex_pts[:, 1]))
    h_loc = _target_spacing(hex_pts[:, 0], hex_pts[:, 1], h_hole, h_outer)
    keep_hex = (r_hex > r_ring_max + 0.55 * h_loc) & (d_outer > 0.52 * h_outer)
    hex_pts = hex_pts[keep_hex]

    pts = np.vstack((outer_pts, hole_pts, ring_pts, hex_pts)).astype(np.float64)
    tris = _filter_plate_tris(pts, Delaunay(pts).simplices)

    # Boundary-locked area-weighted Laplacian / CVT smoothing
    n_pts = len(pts)
    for it in range(smooth_iters):
        areas, _ = _tri_angles_and_areas(pts, tris)
        cent = pts[tris].mean(axis=1)
        w = np.maximum(areas, 1e-14)
        wx = np.zeros(n_pts, dtype=np.float64)
        wy = np.zeros(n_pts, dtype=np.float64)
        wt = np.zeros(n_pts, dtype=np.float64)
        for k in range(3):
            np.add.at(wx, tris[:, k], w * cent[:, 0])
            np.add.at(wy, tris[:, k], w * cent[:, 1])
            np.add.at(wt, tris[:, k], w)
        interior = np.arange(n_bnd, n_pts)
        good = wt[interior] > 1e-15
        idx = interior[good]
        new_x = wx[idx] / wt[idx]
        new_y = wy[idx] / wt[idx]
        pts[idx, 0] = 0.35 * pts[idx, 0] + 0.65 * new_x
        pts[idx, 1] = 0.35 * pts[idx, 1] + 0.65 * new_y
        if (it + 1) % 3 == 0 or it == smooth_iters - 1:
            tris = _filter_plate_tris(pts, Delaunay(pts).simplices)

    # Build topology, edges, boundary sets
    mesh = precompute_p1_mesh(pts, tris, verify=verify)
    return mesh


def precompute_p1_mesh(pts, tris, verify=True):
    """Precompute geometry, P1 strain matrices, boundary edges, and template stiffness matrices."""
    n_pts = len(pts)
    n_tri = len(tris)
    areas, angles_deg = _tri_angles_and_areas(pts, tris)
    assert np.all(areas > 0.0), f"Non-positive triangle area: min={areas.min()}"

    # Edge-to-triangle incidence (detect hanging nodes / cracks / wrong BCs)
    e01 = np.sort(tris[:, [0, 1]], axis=1)
    e12 = np.sort(tris[:, [1, 2]], axis=1)
    e20 = np.sort(tris[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))
    tri_ids = np.tile(np.arange(n_tri, dtype=np.int32), 3)
    # Directed pairs for boundary identification
    dir_edges = np.vstack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))

    edge_keys = all_edges[:, 0].astype(np.int64) * n_pts + all_edges[:, 1].astype(np.int64)
    order = np.argsort(edge_keys, kind="mergesort")
    sorted_keys = edge_keys[order]

    # Group consecutive identical edge keys
    uniq_keys, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    if verify:
        assert np.all((counts == 1) | (counts == 2)), f"Non-manifold/hanging edges detected: counts={np.unique(counts)}"
        min_ang = float(angles_deg.min())
        assert min_ang > 20.0, f"Minimum interior angle too small: {min_ang:.2f} deg"
        exact_area = 1.0 - math.pi * HOLE_R * HOLE_R
        rel_area_err = abs(float(areas.sum()) - exact_area) / exact_area
        assert rel_area_err < 1e-3, f"Mesh area mismatch vs analytical: {rel_area_err:.4e}"
        # Check Euler characteristic V - E + F == 0 for square with 1 circular hole
        chi = n_pts - len(uniq_keys) + n_tri
        assert chi == 0, f"Topological Euler characteristic mismatch (got {chi}, expected 0)"

    # Interior edges (shared by 2 triangles) & Boundary edges (1 triangle)
    int_mask = counts == 2
    int_idx0 = order[starts[int_mask]]
    int_idx1 = order[starts[int_mask] + 1]
    int_edges = all_edges[int_idx0]
    int_t0 = tri_ids[int_idx0]
    int_t1 = tri_ids[int_idx1]
    p_a = pts[int_edges[:, 0]]
    p_b = pts[int_edges[:, 1]]
    vec_int = p_b - p_a
    int_len = np.hypot(vec_int[:, 0], vec_int[:, 1])
    # Unit normal pointing outward from t0? Compute signed area with t0's 3rd vertex or just unit vector perp
    int_normals = np.column_stack((vec_int[:, 1] / int_len, -vec_int[:, 0] / int_len))

    bnd_mask = counts == 1
    bnd_idx = order[starts[bnd_mask]]
    bnd_dir = dir_edges[bnd_idx]  # CCW inside triangle -> outward normal is (dy, -dx)/len
    bnd_tri = tri_ids[bnd_idx]
    pb0 = pts[bnd_dir[:, 0]]
    pb1 = pts[bnd_dir[:, 1]]
    vb = pb1 - pb0
    bnd_len = np.hypot(vb[:, 0], vb[:, 1])
    bnd_normals = np.column_stack((vb[:, 1] / bnd_len, -vb[:, 0] / bnd_len))
    bnd_mid = 0.5 * (pb0 + pb1)

    # Classify boundary edges: left (x=0 Dirichlet), right (x=1 loaded Neumann), free (top/bottom/hole)
    tol = 1e-7
    is_left_e = (np.abs(pb0[:, 0]) < tol) & (np.abs(pb1[:, 0]) < tol)
    is_right_e = (np.abs(pb0[:, 0] - 1.0) < tol) & (np.abs(pb1[:, 0] - 1.0) < tol)
    is_free_e = ~(is_left_e | is_right_e)

    # Dirichlet nodes on x = 0
    is_dirichlet_node = np.abs(pts[:, 0]) < tol
    free_nodes = np.flatnonzero(~is_dirichlet_node).astype(np.int32)
    node_to_free = np.full(n_pts, -1, dtype=np.int32)
    node_to_free[free_nodes] = np.arange(len(free_nodes), dtype=np.int32)

    # Lumped nodal area (integral of phi_j) and triangle centroids
    tri_cent = pts[tris].mean(axis=1)
    nodal_area = np.zeros(n_pts, dtype=np.float64)
    for k in range(3):
        np.add.at(nodal_area, tris[:, k], areas / 3.0)

    # Precompute P1 strain-displacement B_T (n_tri, 3, 6) in Voigt order [eps_xx, eps_yy, gamma_xy]
    x0, y0 = pts[tris[:, 0], 0], pts[tris[:, 0], 1]
    x1, y1 = pts[tris[:, 1], 0], pts[tris[:, 1], 1]
    x2, y2 = pts[tris[:, 2], 0], pts[tris[:, 2], 1]
    inv2a = 1.0 / (2.0 * areas)
    b0 = (y1 - y2) * inv2a
    b1 = (y2 - y0) * inv2a
    b2 = (y0 - y1) * inv2a
    c0 = (x2 - x1) * inv2a
    c1 = (x0 - x2) * inv2a
    c2 = (x1 - x0) * inv2a

    B = np.zeros((n_tri, 3, 6), dtype=np.float64)
    B[:, 0, 0], B[:, 0, 2], B[:, 0, 4] = b0, b1, b2
    B[:, 1, 1], B[:, 1, 3], B[:, 1, 5] = c0, c1, c2
    B[:, 2, 0], B[:, 2, 1] = c0, b0
    B[:, 2, 2], B[:, 2, 3] = c1, b1
    B[:, 2, 4], B[:, 2, 5] = c2, b2

    # Unit constitutive templates:
    # C_mu = diag(2, 2, 1), C_lam = [[1,1,0],[1,1,0],[0,0,0]]
    C_mu = np.diag([2.0, 2.0, 1.0])
    C_lam = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    Ke_mu = np.einsum("tij,ik,tkl->tjl", B, C_mu, B) * areas[:, None, None]
    Ke_lam = np.einsum("tij,ik,tkl->tjl", B, C_lam, B) * areas[:, None, None]

    # Global DOF gather-scatter on all 2*n_pts DOFs & free DOFs
    tri_dofs = np.stack([
        2 * tris[:, 0], 2 * tris[:, 0] + 1,
        2 * tris[:, 1], 2 * tris[:, 1] + 1,
        2 * tris[:, 2], 2 * tris[:, 2] + 1,
    ], axis=1)  # (n_tri, 6)
    free_dofs = np.empty(2 * len(free_nodes), dtype=np.int32)
    free_dofs[0::2] = 2 * free_nodes
    free_dofs[1::2] = 2 * free_nodes + 1

    # Nodal vertex-to-triangle incidence for fast ZZ recovery
    v2t_rows = tris.ravel()
    v2t_cols = np.repeat(np.arange(n_tri, dtype=np.int32), 3)
    v2t_vals = np.repeat(areas, 3)
    v2t_mat = csr_matrix((v2t_vals, (v2t_rows, v2t_cols)), shape=(n_pts, n_tri))

    # Build canonical CSC sparsity pattern once so per-field K_ff assembly takes < 1 ms
    ii = np.repeat(tri_dofs, 6, axis=1).ravel()
    jj = np.tile(tri_dofs, (1, 6)).ravel()
    dof_to_free = np.full(2 * n_pts, -1, dtype=np.int32)
    dof_to_free[free_dofs] = np.arange(len(free_dofs), dtype=np.int32)
    fi = dof_to_free[ii]
    fj = dof_to_free[jj]
    f_mask = (fi >= 0) & (fj >= 0)
    f_rows = fi[f_mask]
    f_cols = fj[f_mask]
    n_fd = len(free_dofs)

    # Deduplicate (f_rows, f_cols) into a sorted CSC template + gather indices
    pair_keys = f_rows.astype(np.int64) * n_fd + f_cols.astype(np.int64)
    uniq_pairs, inv_pair = np.unique(pair_keys, return_inverse=True)
    # Sort by col-major (CSC) order: col = key % n_fd, row = key // n_fd
    u_rows = (uniq_pairs // n_fd).astype(np.int32)
    u_cols = (uniq_pairs % n_fd).astype(np.int32)
    csc_ord = np.lexsort((u_rows, u_cols))
    csc_rows = u_rows[csc_ord]
    csc_cols = u_cols[csc_ord]
    csc_indptr = np.searchsorted(csc_cols, np.arange(n_fd + 1), side="left").astype(np.int32)
    # Map each active local entry to sorted CSC position
    rev_ord = np.empty_like(csc_ord)
    rev_ord[csc_ord] = np.arange(len(csc_ord))
    gather_pos = rev_ord[inv_pair].astype(np.int32)
    csr_indptr = None  # CSC == CSR because matrix is symmetric!

    # Also build 2-DOF free-nodal sparsity pattern for SA-AMG
    n_fn = len(free_nodes)

    return {
        "pts": pts,
        "tris": tris,
        "n_pts": n_pts,
        "n_tri": n_tri,
        "areas": areas,
        "min_angle_deg": float(angles_deg.min()),
        "mean_min_angle_deg": float(angles_deg.min(axis=1).mean()),
        "tri_cent": tri_cent,
        "nodal_area": nodal_area,
        "B": B,
        "Ke_mu": Ke_mu,
        "Ke_lam": Ke_lam,
        "tri_dofs": tri_dofs,
        "is_dirichlet_node": is_dirichlet_node,
        "free_nodes": free_nodes,
        "node_to_free": node_to_free,
        "free_dofs": free_dofs,
        "n_free_nodes": n_fn,
        "n_free_dofs": n_fd,
        "f_mask": f_mask,
        "gather_pos": gather_pos,
        "csc_rows": csc_rows,
        "csc_indptr": csc_indptr,
        "int_edges": int_edges,
        "int_t0": int_t0,
        "int_t1": int_t1,
        "int_len": int_len,
        "int_normals": int_normals,
        "bnd_dir": bnd_dir,
        "bnd_tri": bnd_tri,
        "bnd_len": bnd_len,
        "bnd_normals": bnd_normals,
        "bnd_mid": bnd_mid,
        "is_left_e": is_left_e,
        "is_right_e": is_right_e,
        "is_free_e": is_free_e,
        "v2t_mat": v2t_mat,
    }


def assemble_stiffness_csc(mesh, mu_tri, lam_tri):
    """Fast vectorized assembly of free-DOF symmetric positive-definite stiffness matrix K_ff."""
    Ke = mu_tri[:, None, None] * mesh["Ke_mu"] + lam_tri[:, None, None] * mesh["Ke_lam"]
    vals = Ke.ravel()[mesh["f_mask"]]
    n_fd = mesh["n_free_dofs"]
    data = np.bincount(mesh["gather_pos"], weights=vals, minlength=len(mesh["csc_rows"]))
    K_ff = csr_matrix((data, mesh["csc_rows"], mesh["csc_indptr"]), shape=(n_fd, n_fd))
    return K_ff.tocsc()


def assemble_full_k_matvec(mesh, mu_tri, lam_tri, u_full):
    """Exact full-mesh weak-form K_h u_h (shape (n_pts, 2) or (2*n_pts,)) without Dirichlet elimination."""
    Ke = mu_tri[:, None, None] * mesh["Ke_mu"] + lam_tri[:, None, None] * mesh["Ke_lam"]
    u_vec = u_full.reshape(-1)
    u_el = u_vec[mesh["tri_dofs"]]  # (n_tri, 6)
    f_el = np.einsum("tij,tj->ti", Ke, u_el)
    out = np.zeros(2 * mesh["n_pts"], dtype=np.float64)
    np.add.at(out, mesh["tri_dofs"].ravel(), f_el.ravel())
    return out


# ---------------------------------------------------------------------------
# 2. Random Heterogeneous Moduli, Tractions, Load Assembly & Reference Solver
# ---------------------------------------------------------------------------

def evaluate_log_modulus(coords, weights, modes=12, length=0.15):
    """Evaluate grid-independent cosine Fourier log-Young's modulus field g(x,y) at arbitrary points."""
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    k = np.arange(modes, dtype=np.float64)[None, :]
    px = np.cos(math.pi * x * k)
    py = np.cos(math.pi * y * k)
    px[:, 1:] *= math.sqrt(2.0)
    py[:, 1:] *= math.sqrt(2.0)
    # g = sum_{i,j} w_{ij} px_i py_j
    return np.einsum("ni,ij,nj->n", px, weights, py)


def draw_elasticity_case(rng, sigma_range, nu_range, cfg):
    """Draw one grid-independent random linear-elasticity specification."""
    modes = cfg["moduli_modes"]
    length = cfg["moduli_length"]
    k = np.arange(modes, dtype=np.float64)
    spec = (1.0 + (math.pi * length) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2)) ** (-1.25)
    w = rng.standard_normal((modes, modes)) * spec
    w[0, 0] = 0.0
    sigma_e = float(rng.uniform(*sigma_range))
    w *= sigma_e / max(float(np.linalg.norm(w)), 1e-12)
    nu_p = float(rng.uniform(*nu_range))
    tx_c = np.array([
        rng.uniform(0.6, 1.4),
        rng.uniform(-0.8, 0.8),
        rng.uniform(-0.35, 0.35),
        rng.uniform(-0.35, 0.35),
    ], dtype=np.float64)
    ty_c = np.array([
        rng.uniform(-0.4, 0.4),
        rng.uniform(-0.35, 0.35),
        rng.uniform(-0.35, 0.35),
    ], dtype=np.float64)
    return w, sigma_e, nu_p, tx_c, ty_c


def eval_traction(y, tx_c, ty_c):
    """Right-edge Neumann traction t(y) = (t_x(y), t_y(y))."""
    tx = tx_c[0] + tx_c[1] * (y - 0.5) + tx_c[2] * np.cos(math.pi * y) + tx_c[3] * np.sin(2.0 * math.pi * y)
    ty = ty_c[0] + ty_c[1] * np.sin(math.pi * y) + ty_c[2] * np.cos(2.0 * math.pi * y)
    return tx, ty


def assemble_rhs_right_traction(mesh, tx_c, ty_c):
    """2-point Gauss-Legendre integration of right-edge Neumann traction into global load F_h (2*n_pts,)."""
    right_edges = mesh["bnd_dir"][mesh["is_right_e"]]
    p0 = mesh["pts"][right_edges[:, 0]]
    p1 = mesh["pts"][right_edges[:, 1]]
    le = mesh["bnd_len"][mesh["is_right_e"]]
    # 2-pt Gauss quadrature on [0, 1]: xi in {0.5 +- 1/(2*sqrt(3))}, w = 0.5
    xi1 = 0.5 - 0.5 / math.sqrt(3.0)
    xi2 = 0.5 + 0.5 / math.sqrt(3.0)
    y1 = (1.0 - xi1) * p0[:, 1] + xi1 * p1[:, 1]
    y2 = (1.0 - xi2) * p0[:, 1] + xi2 * p1[:, 1]
    tx1, ty1 = eval_traction(y1, tx_c, ty_c)
    tx2, ty2 = eval_traction(y2, tx_c, ty_c)
    # phi_0(xi) = 1 - xi, phi_1(xi) = xi
    fx0 = 0.5 * le * ((1.0 - xi1) * tx1 + (1.0 - xi2) * tx2)
    fy0 = 0.5 * le * ((1.0 - xi1) * ty1 + (1.0 - xi2) * ty2)
    fx1 = 0.5 * le * (xi1 * tx1 + xi2 * tx2)
    fy1 = 0.5 * le * (xi1 * ty1 + xi2 * ty2)
    F_glob = np.zeros(2 * mesh["n_pts"], dtype=np.float64)
    np.add.at(F_glob, 2 * right_edges[:, 0], fx0)
    np.add.at(F_glob, 2 * right_edges[:, 0] + 1, fy0)
    np.add.at(F_glob, 2 * right_edges[:, 1], fx1)
    np.add.at(F_glob, 2 * right_edges[:, 1] + 1, fy1)
    return F_glob


def case_lame_on_mesh(mesh, w, nu_p, cfg):
    g_tri = evaluate_log_modulus(mesh["tri_cent"], w, cfg["moduli_modes"], cfg["moduli_length"])
    E_tri = np.exp(g_tri)
    mu_tri = E_tri / (2.0 * (1.0 + nu_p))
    lam_tri = (E_tri * nu_p) / ((1.0 + nu_p) * (1.0 - 2.0 * nu_p))
    return E_tri, mu_tri, lam_tri


def solve_elasticity_case(mesh, w, nu_p, tx_c, ty_c, cfg):
    """Assemble and solve exact P1 reference displacement U_h and element von Mises stress."""
    E_tri, mu_tri, lam_tri = case_lame_on_mesh(mesh, w, nu_p, cfg)
    K_ff = assemble_stiffness_csc(mesh, mu_tri, lam_tri)
    F_glob = assemble_rhs_right_traction(mesh, tx_c, ty_c)
    F_f = F_glob[mesh["free_dofs"]]
    u_f = spsolve(K_ff, F_f)
    res_rel = float(np.linalg.norm(K_ff @ u_f - F_f, np.inf) / max(float(np.linalg.norm(F_f, np.inf)), 1e-12))
    U = np.zeros((mesh["n_pts"], 2), dtype=np.float64)
    U.reshape(-1)[mesh["free_dofs"]] = u_f
    return U, res_rel, E_tri, mu_tri, lam_tri, K_ff, F_glob


def compute_element_stress_and_vm(mesh, U, mu_tri, lam_tri, nu_p):
    """Compute piecewise-constant P1 stress [s_xx, s_yy, tau_xy] and exact plane-strain von Mises stress."""
    u_el = U.reshape(-1)[mesh["tri_dofs"]]  # (n_tri, 6)
    eps = np.einsum("tij,tj->ti", mesh["B"], u_el)  # [exx, eyy, gamxy]
    tr = eps[:, 0] + eps[:, 1]
    sxx = 2.0 * mu_tri * eps[:, 0] + lam_tri * tr
    syy = 2.0 * mu_tri * eps[:, 1] + lam_tri * tr
    txy = mu_tri * eps[:, 2]
    szz = nu_p * (sxx + syy)
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * txy * txy
    vm = np.sqrt(np.maximum(vm2, 0.0))
    return np.stack((sxx, syy, txy), axis=1), vm


def linearize_vm_proxy(mesh, sigma_hat, Z_disp, mu_tri, lam_tri, nu_p):
    """Exact Gateaux derivative D sigma_vM(sigma_hat) : sigma(Z_disp) on each triangle."""
    sxx, syy, txy = sigma_hat[:, 0], sigma_hat[:, 1], sigma_hat[:, 2]
    szz = nu_p * (sxx + syy)
    vm = np.sqrt(np.maximum(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * txy * txy, 0.0))
    dsigma_z, _ = compute_element_stress_and_vm(mesh, Z_disp, mu_tri, lam_tri, nu_p)
    # Chain rule including d(szz)/d(sxx)=nu_p, d(szz)/d(syy)=nu_p
    ds_xx = 0.5 * ((2.0 * sxx - syy - szz) + nu_p * (2.0 * szz - sxx - syy))
    ds_yy = 0.5 * ((2.0 * syy - sxx - szz) + nu_p * (2.0 * szz - sxx - syy))
    ds_xy = 3.0 * txy
    num = ds_xx * dsigma_z[:, 0] + ds_yy * dsigma_z[:, 1] + ds_xy * dsigma_z[:, 2]
    return np.where(vm > 1e-12, num / np.maximum(vm, 1e-12), 0.0)


def unit_test_fem_and_vm(mesh_c):
    """Verify FEM Patch Test (< 1e-12) under constant strain + exact Gateaux derivative of von Mises."""
    # Patch test with E = 2.7, nu_P = 0.28 (or heterogeneous constant-strain exact nodal Dirichlet+Neumann check)
    E0, nu0 = 2.7, 0.28
    mu0 = E0 / (2.0 * (1.0 + nu0))
    lam0 = E0 * nu0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
    mu_tri = np.full(mesh_c["n_tri"], mu0)
    lam_tri = np.full(mesh_c["n_tri"], lam0)
    K_ff = assemble_stiffness_csc(mesh_c, mu_tri, lam_tri)

    pts = mesh_c["pts"]
    for label, grad_fn, stress_const in (
        ("uniaxial_strain", lambda x, y: np.column_stack((0.12 * x, np.zeros_like(y))),
         np.array([(2 * mu0 + lam0) * 0.12, lam0 * 0.12, 0.0])),
        ("pure_shear", lambda x, y: np.column_stack((np.zeros_like(x), 0.10 * x)),
         np.array([0.0, 0.0, mu0 * 0.10])),
    ):
        u_exact = grad_fn(pts[:, 0], pts[:, 1])
        # Apply constant traction sigma*n on ALL non-Dirichlet boundary edges (right + top/bottom + hole)
        nd_edges = mesh_c["bnd_dir"][~mesh_c["is_left_e"]]
        normals = mesh_c["bnd_normals"][~mesh_c["is_left_e"]]
        le = mesh_c["bnd_len"][~mesh_c["is_left_e"]]
        tx = stress_const[0] * normals[:, 0] + stress_const[2] * normals[:, 1]
        ty = stress_const[2] * normals[:, 0] + stress_const[1] * normals[:, 1]
        F_pt = np.zeros(2 * mesh_c["n_pts"], dtype=np.float64)
        np.add.at(F_pt, 2 * nd_edges[:, 0], 0.5 * le * tx)
        np.add.at(F_pt, 2 * nd_edges[:, 0] + 1, 0.5 * le * ty)
        np.add.at(F_pt, 2 * nd_edges[:, 1], 0.5 * le * tx)
        np.add.at(F_pt, 2 * nd_edges[:, 1] + 1, 0.5 * le * ty)
        u_sol = np.zeros_like(u_exact)
        u_sol.reshape(-1)[mesh_c["free_dofs"]] = spsolve(K_ff, F_pt[mesh_c["free_dofs"]])
        err = float(np.max(np.abs(u_sol - u_exact)))
        assert err < 1e-12, f"FEM Patch Test ({label}) failed: max_err={err:.3e}"

    # Also test full constant-strain biaxial state (0.12x, -0.05y) with exact Dirichlet lift -K_fd u_D
    u_biax = np.column_stack((0.12 * pts[:, 0], -0.05 * pts[:, 1]))
    s_biax = np.array([2 * mu0 * 0.12 + lam0 * 0.07, 2 * mu0 * (-0.05) + lam0 * 0.07, 0.0])
    nd_edges = mesh_c["bnd_dir"][~mesh_c["is_left_e"]]
    normals = mesh_c["bnd_normals"][~mesh_c["is_left_e"]]
    le = mesh_c["bnd_len"][~mesh_c["is_left_e"]]
    F_bx = np.zeros(2 * mesh_c["n_pts"], dtype=np.float64)
    tx_b = s_biax[0] * normals[:, 0] + s_biax[2] * normals[:, 1]
    ty_b = s_biax[2] * normals[:, 0] + s_biax[1] * normals[:, 1]
    np.add.at(F_bx, 2 * nd_edges[:, 0], 0.5 * le * tx_b)
    np.add.at(F_bx, 2 * nd_edges[:, 0] + 1, 0.5 * le * ty_b)
    np.add.at(F_bx, 2 * nd_edges[:, 1], 0.5 * le * tx_b)
    np.add.at(F_bx, 2 * nd_edges[:, 1] + 1, 0.5 * le * ty_b)
    u_dir_only = np.where(mesh_c["is_dirichlet_node"][:, None], u_biax, 0.0)
    k_u_dir = assemble_full_k_matvec(mesh_c, mu_tri, lam_tri, u_dir_only)
    u_bx_sol = u_dir_only.copy()
    u_bx_sol.reshape(-1)[mesh_c["free_dofs"]] = spsolve(K_ff, (F_bx - k_u_dir)[mesh_c["free_dofs"]])
    assert float(np.max(np.abs(u_bx_sol - u_biax))) < 1e-12, "Biaxial Patch Test failed"

    # Finite-difference check of linearized von Mises Gateaux derivative
    rng = np.random.default_rng(77)
    U0 = rng.normal(scale=0.1, size=(mesh_c["n_pts"], 2))
    Z0 = rng.normal(scale=0.1, size=(mesh_c["n_pts"], 2))
    s_hat, vm_hat = compute_element_stress_and_vm(mesh_c, U0, mu_tri, lam_tri, nu0)
    eps_fd = 1e-7
    _, vm_p = compute_element_stress_and_vm(mesh_c, U0 + eps_fd * Z0, mu_tri, lam_tri, nu0)
    _, vm_m = compute_element_stress_and_vm(mesh_c, U0 - eps_fd * Z0, mu_tri, lam_tri, nu0)
    fd = (vm_p - vm_m) / (2.0 * eps_fd)
    lin = linearize_vm_proxy(mesh_c, s_hat, Z0, mu_tri, lam_tri, nu0)
    vm_err = float(np.max(np.abs(fd - lin)) / np.max(np.abs(fd)))
    assert vm_err < 1e-7, f"von Mises linearization check failed: {vm_err:.3e}"
    print(f"[Task 1 Unit Check] FEM Patch Test (tensile + pure_shear) < 1e-12 & vM Gateaux FD rel_err={vm_err:.2e} PASS", flush=True)


# ---------------------------------------------------------------------------
# 3. Numba-Accelerated 2-DOF Block Smoothed-Aggregation Galerkin AMG (Vector SA-AMG-V2)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _block_greedy_agg(n_nodes, indptr, indices, strength, theta):
    """Two-pass nodal greedy aggregation for 2-DOF block SA-AMG."""
    agg = np.full(n_nodes, -1, dtype=np.int32)
    cur_agg = 0
    # Pass 1: form aggregates around unaggregated seed nodes + strong neighbors
    for i in range(n_nodes):
        if agg[i] != -1:
            continue
        has_free_nb = False
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i and agg[j] == -1 and strength[k] >= theta:
                has_free_nb = True
                break
        if has_free_nb or True:
            agg[i] = cur_agg
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                if j != i and agg[j] == -1 and strength[k] >= theta:
                    agg[j] = cur_agg
            cur_agg += 1
    return agg, cur_agg


@njit(cache=True)
def _extract_2x2_diag_inv(n_nodes, indptr, indices, data):
    """Extract diagonal 2x2 blocks and their exact 2x2 inverses."""
    d_inv = np.zeros((n_nodes, 2, 2), dtype=np.float64)
    for i in range(n_nodes):
        r0 = 2 * i
        r1 = r0 + 1
        a00 = 0.0
        a01 = 0.0
        a10 = 0.0
        a11 = 0.0
        for k in range(indptr[r0], indptr[r0 + 1]):
            c = indices[k]
            if c == r0:
                a00 = data[k]
            elif c == r1:
                a01 = data[k]
        for k in range(indptr[r1], indptr[r1 + 1]):
            c = indices[k]
            if c == r0:
                a10 = data[k]
            elif c == r1:
                a11 = data[k]
        det = a00 * a11 - a01 * a10
        if det <= 1e-30:
            det = 1e-30
        inv_det = 1.0 / det
        d_inv[i, 0, 0] = a11 * inv_det
        d_inv[i, 0, 1] = -a01 * inv_det
        d_inv[i, 1, 0] = -a10 * inv_det
        d_inv[i, 1, 1] = a00 * inv_det
    return d_inv


@njit(cache=True)
def _block_gs_sweep(n_nodes, indptr, indices, data, d_inv, x, b, forward):
    """2x2 nodal block Gauss-Seidel sweep (forward or backward)."""
    for idx in range(n_nodes):
        i = idx if forward else (n_nodes - 1 - idx)
        r0 = 2 * i
        r1 = r0 + 1
        rhs0 = b[r0]
        rhs1 = b[r1]
        for k in range(indptr[r0], indptr[r0 + 1]):
            c = indices[k]
            if c != r0 and c != r1:
                rhs0 -= data[k] * x[c]
        for k in range(indptr[r1], indptr[r1 + 1]):
            c = indices[k]
            if c != r0 and c != r1:
                rhs1 -= data[k] * x[c]
        x[r0] = d_inv[i, 0, 0] * rhs0 + d_inv[i, 0, 1] * rhs1
        x[r1] = d_inv[i, 1, 0] * rhs0 + d_inv[i, 1, 1] * rhs1


def _nodal_strength_graph(K_csr):
    """Build nodal connection graph and normalized block Frobenius strength on CSR K."""
    n_nodes = K_csr.shape[0] // 2
    # Aggregate 2x2 block Frobenius norms via fast COO -> CSR reduction
    K_coo = K_csr.tocoo()
    bi = K_coo.row // 2
    bj = K_coo.col // 2
    pair = bi.astype(np.int64) * n_nodes + bj.astype(np.int64)
    u_pair, inv = np.unique(pair, return_inverse=True)
    frob2 = np.bincount(inv, weights=K_coo.data * K_coo.data, minlength=len(u_pair))
    frob = np.sqrt(frob2)
    u_bi = (u_pair // n_nodes).astype(np.int32)
    u_bj = (u_pair % n_nodes).astype(np.int32)
    diag_f = np.zeros(n_nodes, dtype=np.float64)
    dmask = u_bi == u_bj
    diag_f[u_bi[dmask]] = np.maximum(frob[dmask], 1e-30)
    strength_val = frob / np.sqrt(diag_f[u_bi] * diag_f[u_bj])
    G = csr_matrix((strength_val, (u_bi, u_bj)), shape=(n_nodes, n_nodes))
    G.sort_indices()
    return G


def _build_tentative_block_p(agg, n_nodes, n_agg):
    rows = np.arange(2 * n_nodes, dtype=np.int32)
    cols = np.empty(2 * n_nodes, dtype=np.int32)
    cols[0::2] = 2 * agg
    cols[1::2] = 2 * agg + 1
    vals = np.ones(2 * n_nodes, dtype=np.float64)
    return csr_matrix((vals, (rows, cols)), shape=(2 * n_nodes, 2 * n_agg))


def _estimate_block_jacobi_rho(K_csr, d_inv, iters=12):
    """Estimate spectral radius of D_blk^{-1} K via power iteration."""
    n = K_csr.shape[0]
    rng = np.random.default_rng(12345)
    v = rng.normal(size=n)
    v /= max(float(np.linalg.norm(v)), 1e-30)
    lam = 1.8
    for _ in range(iters):
        kv = K_csr @ v
        # Multiply by d_inv (n_nodes, 2, 2)
        k2 = kv.reshape(-1, 2)
        w = np.einsum("nij,nj->ni", d_inv, k2).reshape(-1)
        nw = float(np.linalg.norm(w))
        if nw <= 1e-30:
            break
        lam = nw
        v = w / nw
    return max(lam, 1.0)


def build_vector_sa_amg(K_csc, min_nodes=60, theta=0.08):
    """Build 2-DOF nodal block Smoothed-Aggregation Galerkin AMG hierarchy."""
    levels = []
    K_cur = K_csc.tocsr()
    K_cur.sort_indices()
    while (K_cur.shape[0] // 2) > min_nodes and len(levels) < 8:
        n_nodes = K_cur.shape[0] // 2
        d_inv = _extract_2x2_diag_inv(n_nodes, K_cur.indptr, K_cur.indices, K_cur.data)
        G = _nodal_strength_graph(K_cur)
        agg, n_agg = _block_greedy_agg(n_nodes, G.indptr, G.indices, G.data, theta)
        if n_agg >= int(0.85 * n_nodes) or n_agg < 4:
            break
        P0 = _build_tentative_block_p(agg, n_nodes, n_agg)
        # Smoothed prolongation: P = (I - omega * D_blk^{-1} K) P0
        rho = _estimate_block_jacobi_rho(K_cur, d_inv, iters=12) * 1.05
        omega = (4.0 / 3.0) / rho
        KP0 = (K_cur @ P0).tocoo()
        # Apply D_blk^{-1} to rows of KP0
        bi = KP0.row // 2
        parity = KP0.row & 1
        # Compute D_blk^{-1} KP0 via dense row-block scaling on CSR entries: convert to dense? No, do sparse:
        row_blk = np.repeat(np.arange(n_nodes, dtype=np.int32), 4)
        # Build sparse D_blk^{-1} CSR explicitly (exact, tiny O(N))
        r_d = np.empty(4 * n_nodes, dtype=np.int32)
        c_d = np.empty(4 * n_nodes, dtype=np.int32)
        idx_n = np.arange(n_nodes, dtype=np.int32)
        r_d[0::4], c_d[0::4] = 2 * idx_n, 2 * idx_n
        r_d[1::4], c_d[1::4] = 2 * idx_n, 2 * idx_n + 1
        r_d[2::4], c_d[2::4] = 2 * idx_n + 1, 2 * idx_n
        r_d[3::4], c_d[3::4] = 2 * idx_n + 1, 2 * idx_n + 1
        Dinv_sp = csr_matrix((d_inv.ravel(), (r_d, c_d)), shape=K_cur.shape)
        P = (P0 - omega * (Dinv_sp @ (K_cur @ P0))).tocsr()
        P.sort_indices()
        PT = P.T.tocsr()
        PT.sort_indices()

        levels.append({
            "K": K_cur,
            "indptr": K_cur.indptr,
            "indices": K_cur.indices,
            "data": K_cur.data,
            "n_nodes": n_nodes,
            "d_inv": d_inv,
            "P": P,
            "PT": PT,
        })
        K_cur = (PT @ K_cur @ P).tocsr()
        K_cur = (0.5 * (K_cur + K_cur.T)).tocsr()
        K_cur.sort_indices()

    levels.append({
        "K_csc": K_cur.tocsc(),
        "K": K_cur,
        "n_nodes": K_cur.shape[0] // 2,
    })
    return levels


def vector_amg_vcycle(levels, b, idx=0, nu=2):
    """Strictly symmetric 2-DOF block SA-AMG V-cycle (nu=2 forward GS -> exact coarse -> nu=2 backward GS)."""
    lvl = levels[idx]
    if idx == len(levels) - 1:
        return spsolve(lvl["K_csc"], b)
    x = np.zeros_like(b)
    n_nodes = lvl["n_nodes"]
    indptr, indices, data, d_inv = lvl["indptr"], lvl["indices"], lvl["data"], lvl["d_inv"]
    for _ in range(nu):
        _block_gs_sweep(n_nodes, indptr, indices, data, d_inv, x, b, True)
    r = b - lvl["K"] @ x
    r_c = lvl["PT"] @ r
    e_c = vector_amg_vcycle(levels, r_c, idx=idx + 1, nu=nu)
    x = x + lvl["P"] @ e_c
    for _ in range(nu):
        _block_gs_sweep(n_nodes, indptr, indices, data, d_inv, x, b, False)
    return x


def lift_vector_amg_v2(levels, r_free, cycles=2, nu=2):
    """2 V-cycles of symmetric Vector SA-AMG (linear, SPD)."""
    x = np.zeros_like(r_free)
    K0 = levels[0]["K"]
    for _ in range(cycles):
        res = r_free - K0 @ x
        x = x + vector_amg_vcycle(levels, res, idx=0, nu=nu)
    return x


def amg_complexity_vector(levels):
    n_nodes0 = levels[0]["n_nodes"]
    nnz0 = levels[0]["K"].nnz
    tot_nodes = sum(l["n_nodes"] for l in levels)
    tot_nnz = sum(l["K"].nnz for l in levels)
    return float(tot_nodes / n_nodes0), float(tot_nnz / nnz0)


def power_iter_energy_vector(K_csc, apply_b, init_vec, max_iter=80, tol=1e-5):
    """Power iteration on E^2 in K_ff inner product for E = I - B K_ff."""
    v = init_vec.copy()
    kv = K_csc @ v
    vk = float(np.dot(v, kv))
    v = v / math.sqrt(max(vk, 1e-30))
    prev = -1.0
    val = 0.0
    for it in range(max_iter):
        ev = v - apply_b(K_csc @ v)
        kev = K_csc @ ev
        ev_k = float(np.dot(ev, kev))
        val = math.sqrt(max(ev_k, 0.0))
        e2v = ev - apply_b(kev)
        ke2v = K_csc @ e2v
        e2_k = float(np.dot(e2v, ke2v))
        if e2_k <= 1e-30:
            val = 0.0
            break
        v = e2v / math.sqrt(e2_k)
        if it >= 3 and abs(val - prev) <= tol * max(val, 1e-12):
            break
        prev = val
    return float(val)


def unit_test_vector_amg(mesh_c, cfg):
    """Verify SA-AMG symmetry (< 1e-11), Galerkin identity, and near-incompressible energy contraction < 1."""
    # Small mesh symmetry + Galerkin test
    mesh_sm = build_perforated_plate_mesh(0.039, 0.080, n_edge=13, n_hole=32, smooth_iters=15, verify=True)
    rng = np.random.default_rng(4242)
    w, _, nu_p, _, _ = draw_elasticity_case(rng, [0.55, 0.85], [0.40, 0.43], cfg)
    _, mu_s, lam_s = case_lame_on_mesh(mesh_sm, w, nu_p, cfg)
    K_sm = assemble_stiffness_csc(mesh_sm, mu_s, lam_s)
    lv_sm = build_vector_sa_amg(K_sm)

    x = rng.normal(size=K_sm.shape[0])
    y = rng.normal(size=K_sm.shape[0])
    bx = lift_vector_amg_v2(lv_sm, x, cycles=2, nu=2)
    by = lift_vector_amg_v2(lv_sm, y, cycles=2, nu=2)
    sym_err = abs(float(np.dot(x, by)) - float(np.dot(y, bx))) / max(abs(float(np.dot(x, by))), 1e-12)
    assert sym_err < 1e-11, f"Vector SA-AMG-V2 symmetry error too high: {sym_err:.3e}"

    # Contraction check on coarse mesh under severe OOD-Lame (nu=0.43, sigma=0.80)
    _, mu_c, lam_c = case_lame_on_mesh(mesh_c, w, 0.43, cfg)
    K_c = assemble_stiffness_csc(mesh_c, mu_c, lam_c)
    lv_c = build_vector_sa_amg(K_c)
    init = rng.normal(size=K_c.shape[0])
    en_norm = power_iter_energy_vector(K_c, lambda r: lift_vector_amg_v2(lv_c, r, cycles=2, nu=2), init)
    assert en_norm < 1.0, f"Vector SA-AMG-V2 failed energy contraction at nu=0.43: {en_norm:.4f}"
    cg, co = amg_complexity_vector(lv_c)
    print(f"[Task 1 Unit Check] Vector SA-AMG-V2 sym_err={sym_err:.2e}, OOD nu=0.43 ||I-BK||_K={en_norm:.4f} < 1, C_G={cg:.3f}, C_O={co:.3f} PASS", flush=True)


# ---------------------------------------------------------------------------
# 4. Unstructured Nodal & Element Physical Envelopes + Classical FEM Baselines
# ---------------------------------------------------------------------------

def build_mass_envelope_matrix(coords, weights, l_phys, cutoff_mult=3.5):
    """Sparse row-normalized mass-weighted Gaussian envelope W_{l,h} via cKDTree."""
    n = len(coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff_mult * l_phys, output_type="ndarray")
    rows = np.concatenate((np.arange(n, dtype=np.int32), pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((np.arange(n, dtype=np.int32), pairs[:, 1], pairs[:, 0]))
    d2 = np.sum((coords[rows] - coords[cols]) ** 2, axis=1)
    raw = weights[cols] * np.exp(-0.5 * d2 / (l_phys * l_phys))
    W = csr_matrix((raw, (rows, cols)), shape=(n, n))
    row_sums = np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1e-30)
    W = csr_matrix((W.data / row_sums[W.nonzero()[0]], W.indices, W.indptr), shape=(n, n))
    W.sort_indices()
    return W


def apply_unstructured_scale(W_mat, weights, val, cfg, free_idx=None):
    """Compute physical-radius mass-weighted RMS envelope + relative floor + quadratic mixture.

    val has shape (N, M) or (N, M, C). If free_idx is provided, global RMS & mixture are computed
    on free coordinates (excluding homogeneous Dirichlet boundary zeros).
    """
    w_tot = weights / np.maximum(weights.sum(), 1e-30)
    rho_rel = cfg["relative_floor"]
    eps_abs = cfg["abs_floor"]
    lam = cfg["lambda"]

    if val.ndim == 2:
        sq = val * val
        smooth = (W_mat @ sq.T).T
        if free_idx is not None:
            wf = weights[free_idx]
            wf = wf / np.maximum(wf.sum(), 1e-30)
            rms2 = (sq[:, free_idx] * wf[None, :]).sum(axis=1, keepdims=True)
        else:
            rms2 = (sq * w_tot[None, :]).sum(axis=1, keepdims=True)
        env2 = smooth + (rho_rel * rho_rel) * rms2 + eps_abs * eps_abs
        if free_idx is not None:
            env_rms2 = (env2[:, free_idx] * wf[None, :]).sum(axis=1, keepdims=True)
        else:
            env_rms2 = (env2 * w_tot[None, :]).sum(axis=1, keepdims=True)
        sig = np.sqrt((1.0 - lam) * env_rms2 + lam * env2)
        return np.maximum(sig, eps_abs)
    else:
        out = np.empty_like(val, dtype=np.float64)
        for c in range(val.shape[2]):
            out[:, :, c] = apply_unstructured_scale(W_mat, weights, val[:, :, c], cfg, free_idx=free_idx)
        return out


def compute_fem_residual_br_and_zz(mesh, U_hat, mu_tri, lam_tri, nu_p, tx_c, ty_c):
    """Compute classical FEM element indicators:
      1) Babuska-Rheinboldt / Verfurth residual indicator (interior traction jumps + Neumann edge error),
      2) Zienkiewicz-Zhu (ZZ) nodal area-averaged continuous stress recovery difference.
    Returns nodal projections (n_pts,) and element indicators (n_tri,).
    """
    sig_hat, _ = compute_element_stress_and_vm(mesh, U_hat, mu_tri, lam_tri, nu_p)  # (n_tri, 3): sxx, syy, txy

    # --- 1) BR / Verfurth element indicator ---
    eta2_tri = np.zeros(mesh["n_tri"], dtype=np.float64)
    # Interior edges: jump [[sigma n_e]]
    t0, t1 = mesh["int_t0"], mesh["int_t1"]
    n_int = mesh["int_normals"]
    he = mesh["int_len"]
    ds_xx = sig_hat[t0, 0] - sig_hat[t1, 0]
    ds_yy = sig_hat[t0, 1] - sig_hat[t1, 1]
    ds_xy = sig_hat[t0, 2] - sig_hat[t1, 2]
    jx = ds_xx * n_int[:, 0] + ds_xy * n_int[:, 1]
    jy = ds_xy * n_int[:, 0] + ds_yy * n_int[:, 1]
    jump2 = (he * he) * (jx * jx + jy * jy)  # h_e * ||jump||_{L2(e)}^2 = h_e^2 |jump|^2
    np.add.at(eta2_tri, t0, 0.5 * jump2)
    np.add.at(eta2_tri, t1, 0.5 * jump2)

    # Neumann edges (right + free)
    # Right edge with 2-pt Gauss quadrature
    re_mask = mesh["is_right_e"]
    re_tri = mesh["bnd_tri"][re_mask]
    re_dir = mesh["bnd_dir"][re_mask]
    re_len = mesh["bnd_len"][re_mask]
    re_n = mesh["bnd_normals"][re_mask]
    p0 = mesh["pts"][re_dir[:, 0]]
    p1 = mesh["pts"][re_dir[:, 1]]
    xi1 = 0.5 - 0.5 / math.sqrt(3.0)
    xi2 = 0.5 + 0.5 / math.sqrt(3.0)
    y1 = (1.0 - xi1) * p0[:, 1] + xi1 * p1[:, 1]
    y2 = (1.0 - xi2) * p0[:, 1] + xi2 * p1[:, 1]
    tx1, ty1 = eval_traction(y1, tx_c, ty_c)
    tx2, ty2 = eval_traction(y2, tx_c, ty_c)
    sn_x = sig_hat[re_tri, 0] * re_n[:, 0] + sig_hat[re_tri, 2] * re_n[:, 1]
    sn_y = sig_hat[re_tri, 2] * re_n[:, 0] + sig_hat[re_tri, 1] * re_n[:, 1]
    r_right = 0.5 * (re_len * re_len) * (((tx1 - sn_x) ** 2 + (ty1 - sn_y) ** 2) + ((tx2 - sn_x) ** 2 + (ty2 - sn_y) ** 2))
    np.add.at(eta2_tri, re_tri, r_right)

    # Free edges (traction = 0)
    fe_mask = mesh["is_free_e"]
    fe_tri = mesh["bnd_tri"][fe_mask]
    fe_len = mesh["bnd_len"][fe_mask]
    fe_n = mesh["bnd_normals"][fe_mask]
    fn_x = sig_hat[fe_tri, 0] * fe_n[:, 0] + sig_hat[fe_tri, 2] * fe_n[:, 1]
    fn_y = sig_hat[fe_tri, 2] * fe_n[:, 0] + sig_hat[fe_tri, 1] * fe_n[:, 1]
    np.add.at(eta2_tri, fe_tri, (fe_len * fe_len) * (fn_x * fn_x + fn_y * fn_y))

    eta_br_tri = np.sqrt(np.maximum(eta2_tri, 0.0))

    # --- 2) Zienkiewicz-Zhu (ZZ) recovered continuous stress indicator ---
    v2t = mesh["v2t_mat"]
    denom_a = np.maximum(np.asarray(v2t.sum(axis=1)).ravel(), 1e-30)
    sig_star = (v2t @ sig_hat) / denom_a[:, None]  # (n_pts, 3)
    tris = mesh["tris"]
    areas = mesh["areas"]
    diff_sq = np.zeros(mesh["n_tri"], dtype=np.float64)
    for k in range(3):
        dv = sig_star[tris[:, k]] - sig_hat
        diff_sq += dv[:, 0] ** 2 + dv[:, 1] ** 2 + 2.0 * (dv[:, 2] ** 2)
    eta_zz_tri = np.sqrt(np.maximum((areas / 3.0) * diff_sq, 0.0))

    # Area-weighted nodal projection for displacement-level scale
    eta_br_node = (v2t @ eta_br_tri) / denom_a
    eta_zz_node = (v2t @ eta_zz_tri) / denom_a
    return eta_br_node, eta_br_tri, eta_zz_node, eta_zz_tri


# ---------------------------------------------------------------------------
# 5. Mesh-Independent Vector DeepONet
# ---------------------------------------------------------------------------

def make_branch_sensors(n_grid=10, n_right=16):
    """Fixed mesh-independent sensor locations inside Omega + right boundary."""
    g = np.linspace(0.05, 0.95, n_grid)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack((gx.ravel(), gy.ravel()))
    inside = np.hypot(pts[:, 0] - HOLE_CX, pts[:, 1] - HOLE_CY) > HOLE_R + 0.015
    e_sensors = pts[inside]  # ~88 fixed sensors inside Omega
    y_right = np.linspace(0.02, 0.98, n_right)
    return e_sensors.astype(np.float64), y_right.astype(np.float64)


def trunk_features(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    dx = x - HOLE_CX
    dy = y - HOLE_CY
    r = np.hypot(dx, dy)
    dist_hole = r - HOLE_R
    cos_t = dx / np.maximum(r, 1e-8)
    sin_t = dy / np.maximum(r, 1e-8)
    return np.column_stack((
        x, y, dist_hole, cos_t, sin_t,
        np.exp(-3.0 * np.maximum(dist_hole, 0.0)),
        x * (1.0 - x),
    )).astype(np.float32)


def branch_features_batch(w_batch, nu_batch, txc_batch, tyc_batch, e_sens, y_right, cfg):
    n = len(w_batch)
    modes, length = cfg["moduli_modes"], cfg["moduli_length"]
    k = np.arange(modes, dtype=np.float64)[None, :]
    px = np.cos(math.pi * e_sens[:, 0:1] * k)
    py = np.cos(math.pi * e_sens[:, 1:2] * k)
    px[:, 1:] *= math.sqrt(2.0)
    py[:, 1:] *= math.sqrt(2.0)
    # g_all: (n, n_sens)
    g_all = np.einsum("si,nij,sj->ns", px, w_batch, py)
    tx = txc_batch[:, 0:1] + txc_batch[:, 1:2] * (y_right[None, :] - 0.5) + \
         txc_batch[:, 2:3] * np.cos(math.pi * y_right[None, :]) + \
         txc_batch[:, 3:4] * np.sin(2.0 * math.pi * y_right[None, :])
    ty = tyc_batch[:, 0:1] + tyc_batch[:, 1:2] * np.sin(math.pi * y_right[None, :]) + \
         tyc_batch[:, 2:3] * np.cos(2.0 * math.pi * y_right[None, :])
    return np.concatenate((g_all, nu_batch[:, None], tx, ty), axis=1).astype(np.float32)


class VectorDeepONet(nn.Module):
    """Mesh-independent 2-component DeepONet with exact x=0 Dirichlet hard-mask."""

    def __init__(self, d_branch, d_trunk=7, wb=256, wt=192, latent=96):
        super().__init__()
        self.latent = latent
        self.branch = nn.Sequential(
            nn.Linear(d_branch, wb), nn.GELU(),
            nn.Linear(wb, wb), nn.GELU(),
            nn.Linear(wb, wb), nn.GELU(),
            nn.Linear(wb, 2 * latent),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d_trunk, wt), nn.GELU(),
            nn.Linear(wt, wt), nn.GELU(),
            nn.Linear(wt, wt), nn.GELU(),
            nn.Linear(wt, 2 * latent),
        )
        self.bias = nn.Parameter(torch.zeros(2))
        self.register_buffer("b_mean", torch.zeros(d_branch))
        self.register_buffer("b_std", torch.ones(d_branch))
        self.register_buffer("u_scale", torch.ones(2))

    def forward(self, b_inp, t_feat, x_coord):
        bn = (b_inp - self.b_mean) / self.b_std
        bo = self.branch(bn).view(-1, 2, self.latent)  # (B, 2, L)
        to = self.trunk(t_feat).view(-1, 2, self.latent)  # (N_pts, 2, L)
        raw = torch.einsum("bcl,ncl->bnc", bo, to) + self.bias[None, None, :]
        # Hard Dirichlet boundary condition u(0, y) = 0 via exact x multiplier
        return raw * x_coord[None, :, None] * self.u_scale[None, None, :]


# ---------------------------------------------------------------------------
# 6. Parallel Dataset Generation & Parallel Post-Processing Workers
# ---------------------------------------------------------------------------

_WORK_MESH_C = None
_WORK_MESH_F = None
_WORK_CFG = None


def _init_worker(mesh_c, mesh_f, cfg):
    global _WORK_MESH_C, _WORK_MESH_F, _WORK_CFG
    _WORK_MESH_C = mesh_c
    _WORK_MESH_F = mesh_f
    _WORK_CFG = cfg


def _gen_one_elasticity(job):
    w, nu_p, tx_c, ty_c, need_fine = job
    Uc, res_c, _, _, _, _, _ = solve_elasticity_case(_WORK_MESH_C, w, nu_p, tx_c, ty_c, _WORK_CFG)
    out = (Uc.astype(np.float32), res_c)
    if need_fine and _WORK_MESH_F is not None:
        Uf, res_f, _, _, _, _, _ = solve_elasticity_case(_WORK_MESH_F, w, nu_p, tx_c, ty_c, _WORK_CFG)
        out = out + (Uf.astype(np.float32), max(res_c, res_f))
    return out


def ensure_elasticity_data(seed, mesh_c, mesh_f, cfg, workers=6):
    sroot = CACHE / f"seed_{seed}" / "data"
    sroot.mkdir(parents=True, exist_ok=True)
    e_sens, y_right = make_branch_sensors()
    out_paths = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mesh_c, mesh_f, cfg)) as pool:
        for split, count in cfg["counts"].items():
            path = sroot / f"{split}.npz"
            out_paths[split] = path
            if path.exists():
                continue
            stream = cfg["streams"][split]
            rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
            sig_r = cfg["sigma_log_e_ood"] if split == "ood" else cfg["sigma_log_e_id"]
            nu_r = cfg["nu_ood"] if split == "ood" else cfg["nu_id"]
            specs = [draw_elasticity_case(rng, sig_r, nu_r, cfg) for _ in range(count)]
            ws = np.stack([sp[0] for sp in specs])
            sigmas = np.array([sp[1] for sp in specs], dtype=np.float64)
            nus = np.array([sp[2] for sp in specs], dtype=np.float64)
            txcs = np.stack([sp[3] for sp in specs])
            tycs = np.stack([sp[4] for sp in specs])
            bf = branch_features_batch(ws, nus, txcs, tycs, e_sens, y_right, cfg)

            need_fine = (split == "paired")
            jobs = [(ws[i], float(nus[i]), txcs[i], tycs[i], need_fine) for i in range(count)]
            entries = list(pool.map(_gen_one_elasticity, jobs, chunksize=8))
            U_c = np.stack([e[0] for e in entries])
            if need_fine:
                U_f = np.stack([e[2] for e in entries])
                max_res = max(float(e[3]) for e in entries)
            else:
                U_f = None
                max_res = max(float(e[1]) for e in entries)
            assert max_res < 1e-11, f"FEM reference solve residual too high ({split}): {max_res:.2e}"
            ids = stream * 10000 + np.arange(count, dtype=np.int64)
            payload = dict(
                id=ids, w=ws.astype(np.float32), sigma_log_e=sigmas.astype(np.float32),
                nu=nus.astype(np.float32), tx_c=txcs.astype(np.float32), ty_c=tycs.astype(np.float32),
                branch=bf, U_c=U_c, max_ref_res=np.array(max_res),
            )
            if U_f is not None:
                payload["U_f"] = U_f
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **payload)
            tmp.replace(path)
            print(f"[Elast Data] seed={seed} {split:6s}: n={count}, max_ref_res={max_res:.2e}", flush=True)
    return out_paths


def train_vector_deeponet(seed, init_seed, data_paths, mesh_c, cfg, device, ckpt_path):
    d_branch = 121
    with np.load(data_paths["train"]) as tr:
        d_branch = tr["branch"].shape[1]
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        mdl = VectorDeepONet(d_branch, 7, cfg["model"]["width_branch"], cfg["model"]["width_trunk"], cfg["model"]["latent_dim"]).to(device)
        mdl.load_state_dict(ck["model"])
        return mdl.eval(), float(ck["train_seconds"]), float(ck["val_rel_l2"])

    with np.load(data_paths["train"]) as tr, np.load(data_paths["val"]) as va:
        tr_b = torch.from_numpy(tr["branch"]).to(device)
        tr_u = torch.from_numpy(tr["U_c"]).to(device)
        va_b = torch.from_numpy(va["branch"]).to(device)
        va_u = torch.from_numpy(va["U_c"]).to(device)

    t_feat = torch.from_numpy(trunk_features(mesh_c["pts"])).to(device)
    x_coord = torch.from_numpy(mesh_c["pts"][:, 0].astype(np.float32)).to(device)

    torch.manual_seed(init_seed)
    mcfg = cfg["model"]
    mdl = VectorDeepONet(d_branch, 7, mcfg["width_branch"], mcfg["width_trunk"], mcfg["latent_dim"]).to(device)
    mdl.b_mean.copy_(tr_b.mean(dim=0))
    mdl.b_std.copy_(tr_b.std(dim=0).clamp_min(1e-6))
    mdl.u_scale.copy_(tr_u.square().mean(dim=(0, 1)).sqrt().clamp_min(1e-6))

    opt = torch.optim.AdamW(mdl.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, mcfg["epochs"], eta_min=3e-5)
    rng = torch.Generator(device="cpu").manual_seed(init_seed)
    best_val, best_state = float("inf"), None
    t0 = perf_counter()

    def _rel_l2(p, y):
        diff = (p - y).flatten(1).norm(dim=1)
        den = y.flatten(1).norm(dim=1).clamp_min(1e-12)
        return (diff / den).mean()

    for ep in range(mcfg["epochs"]):
        mdl.train()
        for ids in torch.randperm(len(tr_b), generator=rng).split(mcfg["batch_size"]):
            opt.zero_grad(set_to_none=True)
            pred = mdl(tr_b[ids], t_feat, x_coord)
            loss = _rel_l2(pred, tr_u[ids])
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == mcfg["epochs"] - 1:
            mdl.eval()
            with torch.inference_mode():
                vp = torch.cat([mdl(b, t_feat, x_coord) for b in va_b.split(128)], dim=0)
                vloss = float(_rel_l2(vp, va_u))
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}

    elapsed = perf_counter() - t0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(ckpt_path, {"model": best_state, "val_rel_l2": best_val, "train_seconds": elapsed})
    mdl.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return mdl.eval(), elapsed, best_val


@torch.inference_mode()
def predict_deeponet(mdl, branch_np, mesh, device, batch=128):
    t_feat = torch.from_numpy(trunk_features(mesh["pts"])).to(device)
    x_coord = torch.from_numpy(mesh["pts"][:, 0].astype(np.float32)).to(device)
    parts = []
    for i in range(0, len(branch_np), batch):
        b = torch.from_numpy(branch_np[i:i + batch]).to(device)
        parts.append(mdl(b, t_feat, x_coord).cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# 7. Parallel Physics Residual, SA-AMG-V2, Exact Oracle, BR & ZZ Worker
# ---------------------------------------------------------------------------

def _process_one_field_worker(job):
    w, nu_p, tx_c, ty_c, u_hat, use_fine, need_oracle = job
    mesh = _WORK_MESH_F if use_fine else _WORK_MESH_C
    _, mu_tri, lam_tri = case_lame_on_mesh(mesh, w, nu_p, _WORK_CFG)
    K_ff = assemble_stiffness_csc(mesh, mu_tri, lam_tri)
    F_glob = assemble_rhs_right_traction(mesh, tx_c, ty_c)
    # Full weak-form residual R_h = K_h U_hat - F_h; u_hat has 0 on Dirichlet nodes
    R_full = assemble_full_k_matvec(mesh, mu_tri, lam_tri, u_hat) - F_glob
    r_f = R_full[mesh["free_dofs"]]

    lv = build_vector_sa_amg(K_ff)
    z_amg_f = lift_vector_amg_v2(lv, r_f, cycles=2, nu=2)
    z_uns_f = lift_vector_amg_v2(lv, np.abs(r_f), cycles=2, nu=2)

    Z_amg = np.zeros((mesh["n_pts"], 2), dtype=np.float64)
    Z_amg.reshape(-1)[mesh["free_dofs"]] = z_amg_f
    Z_uns = np.zeros((mesh["n_pts"], 2), dtype=np.float64)
    Z_uns.reshape(-1)[mesh["free_dofs"]] = z_uns_f
    R_node = np.zeros((mesh["n_pts"], 2), dtype=np.float64)
    R_node.reshape(-1)[mesh["free_dofs"]] = r_f

    if need_oracle:
        z_ex_f = spsolve(K_ff, r_f)
        Z_ex = np.zeros((mesh["n_pts"], 2), dtype=np.float64)
        Z_ex.reshape(-1)[mesh["free_dofs"]] = z_ex_f
    else:
        Z_ex = None

    # Derived von Mises stress & linearized proxy + classical BR & ZZ indicators
    sig_hat, vm_hat = compute_element_stress_and_vm(mesh, u_hat, mu_tri, lam_tri, nu_p)
    z_vm_amg = linearize_vm_proxy(mesh, sig_hat, Z_amg, mu_tri, lam_tri, nu_p)
    z_vm_uns = linearize_vm_proxy(mesh, sig_hat, Z_uns, mu_tri, lam_tri, nu_p)
    z_vm_ex = linearize_vm_proxy(mesh, sig_hat, Z_ex, mu_tri, lam_tri, nu_p) if Z_ex is not None else None
    br_node, br_tri, zz_node, zz_tri = compute_fem_residual_br_and_zz(mesh, u_hat, mu_tri, lam_tri, nu_p, tx_c, ty_c)

    return (
        Z_amg.astype(np.float32),
        Z_uns.astype(np.float32),
        R_node.astype(np.float32),
        Z_ex.astype(np.float32) if Z_ex is not None else None,
        vm_hat.astype(np.float32),
        z_vm_amg.astype(np.float32),
        z_vm_uns.astype(np.float32),
        z_vm_ex.astype(np.float32) if z_vm_ex is not None else None,
        br_node.astype(np.float32),
        br_tri.astype(np.float32),
        zz_node.astype(np.float32),
        zz_tri.astype(np.float32),
        mu_tri.astype(np.float32),
        lam_tri.astype(np.float32),
    )


def process_split_quantities(pool, data, pred_single, use_fine, need_oracle=True):
    n = len(pred_single)
    jobs = [
        (data["w"][i].astype(np.float64), float(data["nu"][i]),
         data["tx_c"][i].astype(np.float64), data["ty_c"][i].astype(np.float64),
         pred_single[i].astype(np.float64), use_fine, need_oracle)
        for i in range(n)
    ]
    res = list(pool.map(_process_one_field_worker, jobs, chunksize=8))
    cols = list(zip(*res))
    out = {
        "Z_amg": np.stack(cols[0]).astype(np.float64),
        "Z_uns": np.stack(cols[1]).astype(np.float64),
        "R_node": np.stack(cols[2]).astype(np.float64),
        "Z_ex": np.stack(cols[3]).astype(np.float64) if need_oracle else None,
        "vm_hat": np.stack(cols[4]).astype(np.float64),
        "z_vm_amg": np.stack(cols[5]).astype(np.float64),
        "z_vm_uns": np.stack(cols[6]).astype(np.float64),
        "z_vm_ex": np.stack(cols[7]).astype(np.float64) if need_oracle else None,
        "br_node": np.stack(cols[8]).astype(np.float64),
        "br_tri": np.stack(cols[9]).astype(np.float64),
        "zz_node": np.stack(cols[10]).astype(np.float64),
        "zz_tri": np.stack(cols[11]).astype(np.float64),
        "mu_tri": np.stack(cols[12]).astype(np.float64),
        "lam_tri": np.stack(cols[13]).astype(np.float64),
    }
    return out


# ---------------------------------------------------------------------------
# 8. Multiscale Vector SA-AMG O(M) Scaling & Seed Energy Contraction Audit
# ---------------------------------------------------------------------------

def run_multiscale_amg_benchmark(cfg):
    print("[Elast Scaling] Pure-operator Vector SA-AMG-V2 multiscale scaling across 5 meshes...", flush=True)
    # Construct 5 mesh levels from ~1.2k DOFs to ~40k+ DOFs
    specs = (
        ("M1", 0.025, 0.055, 18, 50),
        ("M2", 0.0165, 0.037, 28, 76),
        ("M3", 0.0095, 0.022, 46, 132),
        ("M4", 0.0055, 0.013, 78, 228),
        ("M5", 0.0036, 0.0086, 116, 350),
    )
    rows = []
    for name, hh, ho, ne, nh in specs:
        mesh = build_perforated_plate_mesh(hh, ho, ne, nh, smooth_iters=20, verify=True)
        t_set, t_sol, t_tot, t_sp, cgs, cos, c_ens = [], [], [], [], [], [], []
        # Warmup numba
        rng_w = np.random.default_rng(7070)
        w0, _, nu0, txc0, tyc0 = draw_elasticity_case(rng_w, cfg["sigma_log_e_id"], cfg["nu_id"], cfg)
        _, mu0, lam0 = case_lame_on_mesh(mesh, w0, nu0, cfg)
        K0 = assemble_stiffness_csc(mesh, mu0, lam0)
        F0 = assemble_rhs_right_traction(mesh, txc0, tyc0)[mesh["free_dofs"]]
        lv0 = build_vector_sa_amg(K0)
        _ = lift_vector_amg_v2(lv0, F0, cycles=2, nu=2)
        _ = spsolve(K0, F0)

        for idx, seed in enumerate(SEEDS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 9650, idx]))
            w, _, nu_p, tx_c, ty_c = draw_elasticity_case(rng, [0.40, 0.55], [0.25, 0.34], cfg)
            _, mu_t, lam_t = case_lame_on_mesh(mesh, w, nu_p, cfg)
            K = assemble_stiffness_csc(mesh, mu_t, lam_t)
            F_f = assemble_rhs_right_traction(mesh, tx_c, ty_c)[mesh["free_dofs"]]

            t0 = perf_counter_ns()
            lv = build_vector_sa_amg(K)
            ts = (perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            z = lift_vector_amg_v2(lv, F_f, cycles=2, nu=2)
            tv = (perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            u_ex = spsolve(K, F_f)
            te = (perf_counter_ns() - t0) / 1e6

            cg, co = amg_complexity_vector(lv)
            cgs.append(cg)
            cos.append(co)
            t_set.append(ts)
            t_sol.append(tv)
            t_tot.append(ts + tv)
            t_sp.append(te)
            if idx == 0:
                init = rng.normal(size=K.shape[0])
                en_n = power_iter_energy_vector(K, lambda r: lift_vector_amg_v2(lv, r, cycles=2, nu=2), init)
                c_ens.append(en_n)

        row = {
            "label": name,
            "n_pts": mesh["n_pts"],
            "n_tri": mesh["n_tri"],
            "free_dofs": mesh["n_free_dofs"],
            "min_angle_deg": mesh["min_angle_deg"],
            "grid_complexity_CG": float(np.mean(cgs)),
            "operator_complexity_CO": float(np.mean(cos)),
            "amg_setup_p50_ms": float(np.median(t_set)),
            "amg_v2_solve_p50_ms": float(np.median(t_sol)),
            "amg_v2_total_p50_ms": float(np.median(t_tot)),
            "spsolve_p50_ms": float(np.median(t_sp)),
            "speedup_total": float(np.median(t_sp) / np.median(t_tot)),
            "energy_contraction_seed42": float(c_ens[0]),
        }
        rows.append(row)
        print(f"  {name}: nodes={row['n_pts']:5d}, DOFs={row['free_dofs']:5d}, minAng={row['min_angle_deg']:.1f}deg | "
              f"CG={row['grid_complexity_CG']:.3f}, CO={row['operator_complexity_CO']:.3f} | "
              f"AMG_tot={row['amg_v2_total_p50_ms']:.2f}ms (solve={row['amg_v2_solve_p50_ms']:.2f}ms) vs "
              f"spsolve={row['spsolve_p50_ms']:.2f}ms ({row['speedup_total']:.2f}x) | ||I-BK||_K={row['energy_contraction_seed42']:.3f}", flush=True)

    dofs = np.log([r["free_dofs"] for r in rows])
    sl_solve = float(np.polyfit(dofs, np.log([r["amg_v2_solve_p50_ms"] for r in rows]), 1)[0])
    sl_tot = float(np.polyfit(dofs, np.log([r["amg_v2_total_p50_ms"] for r in rows]), 1)[0])
    sl_sp = float(np.polyfit(dofs, np.log([r["spsolve_p50_ms"] for r in rows]), 1)[0])
    print(f"  [Elast Slopes] AMG_solve={sl_solve:.3f}, AMG_tot={sl_tot:.3f}, spsolve={sl_sp:.3f}", flush=True)
    return {
        "levels": rows,
        "log_log_slopes": {"amg_v2_solve": sl_solve, "amg_v2_total": sl_tot, "spsolve": sl_sp},
    }


# ---------------------------------------------------------------------------
# 9. Evaluation Helpers (Simultaneous Displacement & von Mises Bands + Hole Concentration)
# ---------------------------------------------------------------------------

def _eval_band(err_abs, scale, mean_weights, q90, eval_mask=None, hole_mask=None):
    """Evaluate simultaneous coverage, actual width, matched-90% width, and hole-concentration coverage."""
    n = len(err_abs)
    sc = np.maximum(scale, 1e-12)
    ratio = err_abs / sc
    if eval_mask is not None:
        r_eval = ratio.reshape(n, -1)[:, eval_mask]
        s_eval = sc.reshape(n, -1)[:, eval_mask]
        w_m = mean_weights
    else:
        r_eval = ratio.reshape(n, -1)
        s_eval = sc.reshape(n, -1)
        w_m = mean_weights

    scores = r_eval.max(axis=1)
    w_norm = w_m / np.maximum(w_m.sum(), 1e-30)
    mean_s = (s_eval * w_norm[None, :]).sum(axis=1)
    cov = float((scores <= q90).mean())
    act_w = float((2.0 * q90 * mean_s).mean())
    q_emp = value_at_coverage(scores, 0.90)
    m90_w = float(2.0 * q_emp * mean_s.mean())
    out = {
        "coverage": cov,
        "actual_width": act_w,
        "matched90_width": m90_w,
    }
    if hole_mask is not None:
        r_h = ratio.reshape(n, -1)[:, hole_mask]
        out["hole_zone_cov"] = float((r_h.max(axis=1) <= q90).mean())
    return out


# ---------------------------------------------------------------------------
# 10. Main 5-Seed Pipeline
# ---------------------------------------------------------------------------

def run_task1_elasticity(device=None, workers=6):
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t_all = perf_counter()

    # Build locked coarse and fine unstructured meshes
    mesh_c = build_perforated_plate_mesh(0.0165, 0.037, n_edge=30, n_hole=76, smooth_iters=26, verify=True)
    mesh_f = build_perforated_plate_mesh(0.0085, 0.019, n_edge=60, n_hole=150, smooth_iters=26, verify=True)
    print(f"[Elast Meshes] Coarse T_c: nodes={mesh_c['n_pts']}, tris={mesh_c['n_tri']}, freeDOFs={mesh_c['n_free_dofs']}, "
          f"minAng={mesh_c['min_angle_deg']:.2f}deg | Fine T_f: nodes={mesh_f['n_pts']}, tris={mesh_f['n_tri']}, "
          f"freeDOFs={mesh_f['n_free_dofs']}, minAng={mesh_f['min_angle_deg']:.2f}deg", flush=True)

    unit_test_fem_and_vm(mesh_c)
    unit_test_vector_amg(mesh_c, ELAST_CFG)
    amg_scaling = run_multiscale_amg_benchmark(ELAST_CFG)

    # Masks on coarse & fine meshes
    def _mesh_masks(mesh):
        free_n = mesh["free_nodes"]
        n_pts = mesh["n_pts"]
        r_n = np.hypot(mesh["pts"][:, 0] - HOLE_CX, mesh["pts"][:, 1] - HOLE_CY)
        hole_n = (~mesh["is_dirichlet_node"]) & (r_n <= HOLE_R + 0.15)
        free_dof_eval = mesh["free_dofs"]
        hole_dof_eval = np.empty(2 * int(hole_n.sum()), dtype=np.int32)
        hn_idx = np.flatnonzero(hole_n)
        hole_dof_eval[0::2] = 2 * hn_idx
        hole_dof_eval[1::2] = 2 * hn_idx + 1
        w_free_dof = np.repeat(mesh["nodal_area"][free_n], 2)

        r_t = np.hypot(mesh["tri_cent"][:, 0] - HOLE_CX, mesh["tri_cent"][:, 1] - HOLE_CY)
        hole_t = np.flatnonzero(r_t <= HOLE_R + 0.15).astype(np.int32)
        return {
            "free_n": free_n,
            "free_dof_eval": free_dof_eval,
            "hole_dof_eval": hole_dof_eval,
            "w_free_dof": w_free_dof,
            "hole_t": hole_t,
            "tri_areas": mesh["areas"],
        }

    masks_c = _mesh_masks(mesh_c)
    masks_f = _mesh_masks(mesh_f)

    # Generate datasets across all 5 formal seeds
    data_paths_by_seed = {}
    for seed in SEEDS:
        data_paths_by_seed[seed] = ensure_elasticity_data(seed, mesh_c, mesh_f, ELAST_CFG, workers=workers)

    # Lock l_phys on seed 42 Val only (from {0.03, 0.05, 0.07}; h_hole=0.022 -> l_phys=0.05 gives l_phys/h_hole = 2.27 >= 2)
    s42_root = CACHE / "seed_42"
    m42, _, _ = train_vector_deeponet(42, 42, data_paths_by_seed[42], mesh_c, ELAST_CFG, device, s42_root / "member_0.pt")
    with np.load(data_paths_by_seed[42]["val"]) as va42:
        va_d = {k: va42[k] for k in va42.files}
    va_pred = predict_deeponet(m42, va_d["branch"], mesh_c, device)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mesh_c, mesh_f, ELAST_CFG)) as pool:
        va_q = process_split_quantities(pool, va_d, va_pred, use_fine=False, need_oracle=False)

    # Also lock Raw-|R| PI-CP lambda on seed 42 Val
    va_err = np.abs(va_pred - va_d["U_c"].astype(np.float64))
    va_r_norm = np.abs(va_q["R_node"])
    # Normalize PI-CP scale smoothly with Val median residual amplitude
    r_med = max(float(np.median(va_r_norm[:, masks_c["free_n"]])), 1e-12)
    pi_lam_grid = (0.5 / r_med, 1.0 / r_med, 2.0 / r_med, 5.0 / r_med)
    best_pi_lam = pi_lam_grid[1]
    best_pi_sc = float("inf")
    for pl in pi_lam_grid:
        sc = 1.0 + pl * va_r_norm
        ratio = (va_err.reshape(len(va_err), -1)[:, masks_c["free_dof_eval"]] /
                 sc.reshape(len(va_err), -1)[:, masks_c["free_dof_eval"]])
        q_v = value_at_coverage(ratio.max(axis=1), 0.90)
        w_v = 2.0 * q_v * sc.reshape(len(va_err), -1)[:, masks_c["free_dof_eval"]].mean()
        if w_v < best_pi_sc:
            best_pi_sc = w_v
            best_pi_lam = pl

    best_lphys = 0.05
    best_l_w = float("inf")
    for lp in (0.03, 0.05, 0.07):
        Wn = build_mass_envelope_matrix(mesh_c["pts"], mesh_c["nodal_area"], lp)
        sc = apply_unstructured_scale(Wn, mesh_c["nodal_area"], va_q["Z_amg"], ELAST_CFG, free_idx=masks_c["free_n"])
        ratio = (va_err.reshape(len(va_err), -1)[:, masks_c["free_dof_eval"]] /
                 sc.reshape(len(va_err), -1)[:, masks_c["free_dof_eval"]])
        q_v = value_at_coverage(ratio.max(axis=1), 0.90)
        w_v = 2.0 * q_v * sc.reshape(len(va_err), -1)[:, masks_c["free_dof_eval"]].mean()
        if w_v < best_l_w:
            best_l_w = w_v
            best_lphys = lp
    ELAST_CFG["locked_l_phys"] = best_lphys
    ELAST_CFG["locked_pi_lam"] = float(best_pi_lam)
    print(f"[Elast Val Lock (seed 42)] l_phys={best_lphys:.3f} (l_phys/h_hole={best_lphys/0.022:.2f} >= 2), "
          f"PI-CP lambda={best_pi_lam:.4f}", flush=True)

    # Prebuild sparse envelope matrices once for coarse & fine meshes
    W_node_c = build_mass_envelope_matrix(mesh_c["pts"], mesh_c["nodal_area"], best_lphys)
    W_tri_c = build_mass_envelope_matrix(mesh_c["tri_cent"], mesh_c["areas"], best_lphys)
    W_node_f = build_mass_envelope_matrix(mesh_f["pts"], mesh_f["nodal_area"], best_lphys)
    W_tri_f = build_mass_envelope_matrix(mesh_f["tri_cent"], mesh_f["areas"], best_lphys)

    disp_methods = (
        "Uniform-CP",
        "Raw |R| (PI-CP)",
        "BR/Verfurth FEM + CP",
        "Zienkiewicz-Zhu + CP",
        "Jha Corrector + Unif-CP",
        "Jha Corrector + OpRes-CP",
        "5-Ens. (same center)",
        "5-Ens. (mean center)",
        "Unsigned Vector AMG-V2",
        "OpRes-CP (Vector SA-AMG-V2)",
        "OpRes-CP (Exact K^-1 oracle)",
    )
    vm_methods = (
        "Uniform-CP",
        "BR/Verfurth FEM + CP",
        "Zienkiewicz-Zhu + CP",
        "Jha Corrector + Unif-CP",
        "Jha Corrector + OpRes-CP",
        "5-Ens. (mean center)",
        "Unsigned Vector AMG-V2",
        "OpRes-CP (Vector SA-AMG-V2)",
        "OpRes-CP (Exact K^-1 oracle)",
    )

    per_seed = {}
    for seed in SEEDS:
        paths = data_paths_by_seed[seed]
        sroot = CACHE / f"seed_{seed}"

        # Train 5 DeepONet members (1 single + 4 ensemble companions)
        models, train_times, val_errs = [], [], []
        for m_idx, off in enumerate(ELAST_CFG["ensemble_offsets"]):
            mdl, t_tr, v_l2 = train_vector_deeponet(seed, seed + off, paths, mesh_c, ELAST_CFG, device, sroot / f"member_{m_idx}.pt")
            models.append(mdl)
            train_times.append(t_tr)
            val_errs.append(v_l2)
        print(f"[Elast Train] seed={seed} | member0 val_rel_l2={val_errs[0]:.4f}, single={train_times[0]:.1f}s, "
              f"ens_5={sum(train_times):.1f}s", flush=True)

        # Evaluate ensemble floor on Val
        with np.load(paths["val"]) as va:
            va_br, va_u = va["branch"], va["U_c"].astype(np.float64)
        va_stack = np.stack([predict_deeponet(m, va_br, mesh_c, device) for m in models], axis=0)
        va_std = va_stack.std(axis=0, ddof=1)
        ens_floor = max(float(0.05 * np.median(va_std[:, masks_c["free_n"]])), 1e-8)

        # Helper to compute all method centers, scales, truths, and spatial fidelity metrics for a split
        def _build_split(pool, sp_name, use_fine):
            with np.load(paths[sp_name]) as raw:
                d = {k: raw[k] for k in raw.files}
            mesh = mesh_f if use_fine else mesh_c
            masks = masks_f if use_fine else masks_c
            Wn = W_node_f if use_fine else W_node_c
            Wt = W_tri_f if use_fine else W_tri_c
            u_true = d["U_f" if use_fine else "U_c"].astype(np.float64)

            stack = np.stack([predict_deeponet(m, d["branch"], mesh, device).astype(np.float64) for m in models], axis=0)
            p_single = stack[0]
            p_ens_mean = stack.mean(axis=0)
            p_ens_std = np.hypot(stack.std(axis=0, ddof=1), ens_floor)

            qnt = process_split_quantities(pool, d, p_single, use_fine=use_fine, need_oracle=True)
            # True element von Mises & ensemble mean/std von Mises stress
            n = len(u_true)
            vm_true = np.empty((n, mesh["n_tri"]), dtype=np.float64)
            vm_ens_all = np.empty((len(models), n, mesh["n_tri"]), dtype=np.float64)
            vm_ens_mean = np.empty((n, mesh["n_tri"]), dtype=np.float64)
            vm_jha = np.empty((n, mesh["n_tri"]), dtype=np.float64)
            u_jha = p_single - qnt["Z_amg"]
            for i in range(n):
                mu_t, lam_t, nu_i = qnt["mu_tri"][i], qnt["lam_tri"][i], float(d["nu"][i])
                _, vm_true[i] = compute_element_stress_and_vm(mesh, u_true[i], mu_t, lam_t, nu_i)
                _, vm_jha[i] = compute_element_stress_and_vm(mesh, u_jha[i], mu_t, lam_t, nu_i)
                for m_idx in range(len(models)):
                    _, vm_ens_all[m_idx, i] = compute_element_stress_and_vm(mesh, stack[m_idx, i], mu_t, lam_t, nu_i)
                _, vm_ens_mean[i] = compute_element_stress_and_vm(mesh, p_ens_mean[i], mu_t, lam_t, nu_i)
            vm_ens_std = np.hypot(vm_ens_all.std(axis=0, ddof=1), 1e-6)

            # Jha 2nd-step residual lift for orthogonal compatibility test (Jha Corrector + OpRes-CP)
            qnt_jha = process_split_quantities(pool, d, u_jha, use_fine=use_fine, need_oracle=False)

            # Displacement centers & scales
            ones_n = np.ones_like(p_single)
            s_op_amg = apply_unstructured_scale(Wn, mesh["nodal_area"], qnt["Z_amg"], ELAST_CFG, free_idx=masks["free_n"])
            s_op_ex = apply_unstructured_scale(Wn, mesh["nodal_area"], qnt["Z_ex"], ELAST_CFG, free_idx=masks["free_n"])
            s_op_uns = apply_unstructured_scale(Wn, mesh["nodal_area"], qnt["Z_uns"], ELAST_CFG, free_idx=masks["free_n"])
            s_br_n = np.repeat(apply_unstructured_scale(Wn, mesh["nodal_area"], qnt["br_node"], ELAST_CFG, free_idx=masks["free_n"])[:, :, None], 2, axis=2)
            s_zz_n = np.repeat(apply_unstructured_scale(Wn, mesh["nodal_area"], qnt["zz_node"], ELAST_CFG, free_idx=masks["free_n"])[:, :, None], 2, axis=2)
            s_jha_op = apply_unstructured_scale(Wn, mesh["nodal_area"], qnt_jha["Z_amg"], ELAST_CFG, free_idx=masks["free_n"])

            d_arms = {
                "Uniform-CP": (p_single, ones_n),
                "Raw |R| (PI-CP)": (p_single, 1.0 + best_pi_lam * np.abs(qnt["R_node"])),
                "BR/Verfurth FEM + CP": (p_single, s_br_n),
                "Zienkiewicz-Zhu + CP": (p_single, s_zz_n),
                "Jha Corrector + Unif-CP": (u_jha, ones_n),
                "Jha Corrector + OpRes-CP": (u_jha, s_jha_op),
                "5-Ens. (same center)": (p_single, p_ens_std),
                "5-Ens. (mean center)": (p_ens_mean, p_ens_std),
                "Unsigned Vector AMG-V2": (p_single, s_op_uns),
                "OpRes-CP (Vector SA-AMG-V2)": (p_single, s_op_amg),
                "OpRes-CP (Exact K^-1 oracle)": (p_single, s_op_ex),
            }

            # von Mises stress centers & scales
            ones_t = np.ones_like(vm_true)
            sv_amg = apply_unstructured_scale(Wt, mesh["areas"], qnt["z_vm_amg"], ELAST_CFG)
            sv_ex = apply_unstructured_scale(Wt, mesh["areas"], qnt["z_vm_ex"], ELAST_CFG)
            sv_uns = apply_unstructured_scale(Wt, mesh["areas"], qnt["z_vm_uns"], ELAST_CFG)
            sv_br = apply_unstructured_scale(Wt, mesh["areas"], qnt["br_tri"], ELAST_CFG)
            sv_zz = apply_unstructured_scale(Wt, mesh["areas"], qnt["zz_tri"], ELAST_CFG)
            sv_jha_op = apply_unstructured_scale(Wt, mesh["areas"], qnt_jha["z_vm_amg"], ELAST_CFG)

            v_arms = {
                "Uniform-CP": (qnt["vm_hat"], ones_t),
                "BR/Verfurth FEM + CP": (qnt["vm_hat"], sv_br),
                "Zienkiewicz-Zhu + CP": (qnt["vm_hat"], sv_zz),
                "Jha Corrector + Unif-CP": (vm_jha, ones_t),
                "Jha Corrector + OpRes-CP": (vm_jha, sv_jha_op),
                "5-Ens. (mean center)": (vm_ens_mean, vm_ens_std),
                "Unsigned Vector AMG-V2": (qnt["vm_hat"], sv_uns),
                "OpRes-CP (Vector SA-AMG-V2)": (qnt["vm_hat"], sv_amg),
                "OpRes-CP (Exact K^-1 oracle)": (qnt["vm_hat"], sv_ex),
            }

            # Point relative L2 errors (displacement & von Mises)
            def _rl2(c, y, free_dof=None):
                if free_dof is not None:
                    df = (c - y).reshape(n, -1)[:, free_dof]
                    dn = y.reshape(n, -1)[:, free_dof]
                else:
                    df = (c - y).reshape(n, -1)
                    dn = y.reshape(n, -1)
                return float((np.linalg.norm(df, axis=1) / np.maximum(np.linalg.norm(dn, axis=1), 1e-12)).mean())

            pt_err = {
                "single_disp_rel_l2": _rl2(p_single, u_true, masks["free_dof_eval"]),
                "ens_mean_disp_rel_l2": _rl2(p_ens_mean, u_true, masks["free_dof_eval"]),
                "jha_disp_rel_l2": _rl2(u_jha, u_true, masks["free_dof_eval"]),
                "single_vm_rel_l2": _rl2(qnt["vm_hat"], vm_true),
                "ens_mean_vm_rel_l2": _rl2(vm_ens_mean, vm_true),
                "jha_vm_rel_l2": _rl2(vm_jha, vm_true),
            }

            # Spatial fidelity & signed recovery on free DOFs / elements
            e_disp = (p_single - u_true).reshape(n, -1)[:, masks["free_dof_eval"]]
            z_disp = qnt["Z_amg"].reshape(n, -1)[:, masks["free_dof_eval"]]
            s_disp = s_op_amg.reshape(n, -1)[:, masks["free_dof_eval"]]
            e_vm = qnt["vm_hat"] - vm_true

            fidelity = {
                "disp_signed": compute_signed_recovery(z_disp, e_disp),
                "disp_opres_scale": compute_scale_metrics(np.abs(e_disp), s_disp),
                "disp_ens_scale": compute_scale_metrics(np.abs(e_disp), p_ens_std.reshape(n, -1)[:, masks["free_dof_eval"]]),
                "disp_br_scale": compute_scale_metrics(np.abs(e_disp), s_br_n.reshape(n, -1)[:, masks["free_dof_eval"]]),
                "disp_zz_scale": compute_scale_metrics(np.abs(e_disp), s_zz_n.reshape(n, -1)[:, masks["free_dof_eval"]]),
                "vm_signed": compute_signed_recovery(qnt["z_vm_amg"], e_vm),
                "vm_opres_scale": compute_scale_metrics(np.abs(e_vm), sv_amg),
                "vm_ens_scale": compute_scale_metrics(np.abs(e_vm), vm_ens_std),
                "vm_br_scale": compute_scale_metrics(np.abs(e_vm), sv_br),
                "vm_zz_scale": compute_scale_metrics(np.abs(e_vm), sv_zz),
            }

            # Save seed-42 ID sample 0 for fig10 visualization
            if seed == 42 and sp_name == "id":
                sc_vm_tmp = (np.abs(e_vm) / np.maximum(sv_amg, 1e-12)).max(axis=1)
                q_tmp = quantile(sc_vm_tmp, 0.10)
                np.savez(
                    OUT / "elasticity_example_s42.npz",
                    pts=mesh_c["pts"], tris=mesh_c["tris"],
                    u_true=u_true[0], u_hat=p_single[0],
                    vm_true=vm_true[0], vm_hat=qnt["vm_hat"][0],
                    vm_err=np.abs(e_vm[0]), vm_opres_scale=sv_amg[0],
                    vm_opres_width=(2.0 * q_tmp * sv_amg[0]),
                    disp_err=np.linalg.norm(p_single[0] - u_true[0], axis=1),
                    disp_opres_scale=np.linalg.norm(s_op_amg[0], axis=1),
                )

            return d, mesh, masks, u_true, vm_true, d_arms, v_arms, pt_err, fidelity

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mesh_c, mesh_f, ELAST_CFG)) as pool:
            cal_sp = _build_split(pool, "cal", use_fine=False)
            id_sp = _build_split(pool, "id", use_fine=False)
            fine_sp = _build_split(pool, "paired", use_fine=True)
            ood_sp = _build_split(pool, "ood", use_fine=False)

        # Calibrate q90 on Cal=400
        _, _, cal_masks, cal_u, cal_vm, cal_d_arms, cal_v_arms, _, _ = cal_sp
        q_disp = {}
        for mname in disp_methods:
            c, s = cal_d_arms[mname]
            sc = (np.abs(c - cal_u).reshape(len(cal_u), -1)[:, cal_masks["free_dof_eval"]] /
                  np.maximum(s.reshape(len(cal_u), -1)[:, cal_masks["free_dof_eval"]], 1e-12)).max(axis=1)
            q_disp[mname] = quantile(sc, 0.10)

        q_vm = {}
        for mname in vm_methods:
            c, s = cal_v_arms[mname]
            sc = (np.abs(c - cal_vm) / np.maximum(s, 1e-12)).max(axis=1)
            q_vm[mname] = quantile(sc, 0.10)

        # Evaluate splits
        seed_splits = {}
        for sp_key, sp_pkg in (("id", id_sp), ("fine_paired", fine_sp), ("ood_lame", ood_sp)):
            _, _, sp_masks, y_u, y_vm, sp_d_arms, sp_v_arms, pt_err, fidelity = sp_pkg
            d_res = {}
            for mname in disp_methods:
                c, s = sp_d_arms[mname]
                d_res[mname] = _eval_band(
                    np.abs(c - y_u), s, sp_masks["w_free_dof"], q_disp[mname],
                    eval_mask=sp_masks["free_dof_eval"], hole_mask=sp_masks["hole_dof_eval"],
                )
            v_res = {}
            for mname in vm_methods:
                c, s = sp_v_arms[mname]
                v_res[mname] = _eval_band(
                    np.abs(c - y_vm), s, sp_masks["tri_areas"], q_vm[mname],
                    eval_mask=None, hole_mask=sp_masks["hole_t"],
                )
            seed_splits[sp_key] = {
                "point_error": pt_err,
                "fidelity": fidelity,
                "displacement": d_res,
                "von_mises": v_res,
            }

        # Energy contraction audit on 20 fields/split (ID, Fine, OOD-Lame)
        contr = {}
        for sp_key, sp_pkg, use_f in (("id", id_sp, False), ("fine_paired", fine_sp, True), ("ood_lame", ood_sp, False)):
            d_raw = sp_pkg[0]
            mesh_u = mesh_f if use_f else mesh_c
            en_vals = []
            rng_pi = np.random.default_rng(np.random.SeedSequence([seed, 9699, int(use_f)]))
            for i in range(20):
                _, mu_t, lam_t = case_lame_on_mesh(mesh_u, d_raw["w"][i].astype(np.float64), float(d_raw["nu"][i]), ELAST_CFG)
                K_ff = assemble_stiffness_csc(mesh_u, mu_t, lam_t)
                lv = build_vector_sa_amg(K_ff)
                init = rng_pi.normal(size=K_ff.shape[0])
                en_vals.append(power_iter_energy_vector(K_ff, lambda r: lift_vector_amg_v2(lv, r, cycles=2, nu=2), init))
            contr[sp_key] = {
                "energy_norm_mean": float(np.mean(en_vals)),
                "energy_norm_max": float(np.max(en_vals)),
                "frac_lt_1": float(np.mean([v < 1.0 for v in en_vals])),
            }

        # Single-case P50 latency benchmark (ID coarse T_c & paired fine T_f, 20 timed cases)
        timings = {}
        for t_lbl, d_raw, mesh_u, Wn_u, Wt_u in (
            ("coarse", id_sp[0], mesh_c, W_node_c, W_tri_c),
            ("fine", fine_sp[0], mesh_f, W_node_f, W_tri_f),
        ):
            t_f_s, t_f_e, t_res, t_amg, t_orac, t_env = [], [], [], [], [], []
            t_feat_u = torch.from_numpy(trunk_features(mesh_u["pts"])).to(device)
            x_coord_u = torch.from_numpy(mesh_u["pts"][:, 0].astype(np.float32)).to(device)
            _ = predict_deeponet(models[0], d_raw["branch"][:1], mesh_u, device, batch=1)
            torch.cuda.synchronize(device)
            for i in range(20):
                bi = torch.from_numpy(d_raw["branch"][i:i + 1]).to(device)
                torch.cuda.synchronize(device)
                t0 = perf_counter_ns()
                with torch.inference_mode():
                    u_p = models[0](bi, t_feat_u, x_coord_u)[0].cpu().numpy().astype(np.float64)
                torch.cuda.synchronize(device)
                t_f_s.append((perf_counter_ns() - t0) / 1e6)

                torch.cuda.synchronize(device)
                t0 = perf_counter_ns()
                with torch.inference_mode():
                    for m in models:
                        _ = m(bi, t_feat_u, x_coord_u).cpu().numpy()
                torch.cuda.synchronize(device)
                t_f_e.append((perf_counter_ns() - t0) / 1e6)

                w_i, nu_i = d_raw["w"][i].astype(np.float64), float(d_raw["nu"][i])
                tx_i, ty_i = d_raw["tx_c"][i].astype(np.float64), d_raw["ty_c"][i].astype(np.float64)
                t0 = perf_counter_ns()
                _, mu_t, lam_t = case_lame_on_mesh(mesh_u, w_i, nu_i, ELAST_CFG)
                K_ff = assemble_stiffness_csc(mesh_u, mu_t, lam_t)
                F_g = assemble_rhs_right_traction(mesh_u, tx_i, ty_i)
                r_f = (assemble_full_k_matvec(mesh_u, mu_t, lam_t, u_p) - F_g)[mesh_u["free_dofs"]]
                t_res.append((perf_counter_ns() - t0) / 1e6)

                t0 = perf_counter_ns()
                lv = build_vector_sa_amg(K_ff)
                z_f = lift_vector_amg_v2(lv, r_f, cycles=2, nu=2)
                t_amg.append((perf_counter_ns() - t0) / 1e6)

                t0 = perf_counter_ns()
                _ = spsolve(K_ff, r_f)
                t_orac.append((perf_counter_ns() - t0) / 1e6)

                Z_tmp = np.zeros((mesh_u["n_pts"], 2), dtype=np.float64)
                Z_tmp.reshape(-1)[mesh_u["free_dofs"]] = z_f
                sh, _ = compute_element_stress_and_vm(mesh_u, u_p, mu_t, lam_t, nu_i)
                zvm = linearize_vm_proxy(mesh_u, sh, Z_tmp, mu_t, lam_t, nu_i)
                t0 = perf_counter_ns()
                _ = apply_unstructured_scale(Wn_u, mesh_u["nodal_area"], Z_tmp[None], ELAST_CFG, free_idx=mesh_u["free_nodes"])
                _ = apply_unstructured_scale(Wt_u, mesh_u["areas"], zvm[None], ELAST_CFG)
                t_env.append((perf_counter_ns() - t0) / 1e6)

            timings[t_lbl] = {
                "single_deeponet_ms": float(np.median(t_f_s)),
                "ensemble_5_ms": float(np.median(t_f_e)),
                "fem_res_kff_ms": float(np.median(t_res)),
                "sa_amg_v2_ms": float(np.median(t_amg)),
                "exact_oracle_ms": float(np.median(t_orac)),
                "envelopes_ms": float(np.median(t_env)),
                "opres_total_ms": float(np.median(np.array(t_f_s) + np.array(t_res) + np.array(t_amg) + np.array(t_env))),
            }

        per_seed[str(seed)] = {
            "train_time_single_s": float(train_times[0]),
            "train_time_ens5_s": float(sum(train_times)),
            "val_rel_l2_single": float(val_errs[0]),
            "splits": seed_splits,
            "energy_contraction": contr,
            "timings": timings,
        }

        op_d = seed_splits["id"]["displacement"]["OpRes-CP (Vector SA-AMG-V2)"]
        en_d = seed_splits["id"]["displacement"]["5-Ens. (mean center)"]
        op_v = seed_splits["id"]["von_mises"]["OpRes-CP (Vector SA-AMG-V2)"]
        en_v = seed_splits["id"]["von_mises"]["5-Ens. (mean center)"]
        print(f"  seed={seed} ID Disp: OpRes={100*op_d['coverage']:.1f}%/{op_d['matched90_width']:.5f} vs Ens={100*en_d['coverage']:.1f}%/{en_d['matched90_width']:.5f} | "
              f"vM: OpRes={100*op_v['coverage']:.1f}%/{op_v['matched90_width']:.4f} vs Ens={100*en_v['coverage']:.1f}%/{en_v['matched90_width']:.4f} | "
              f"||I-BK||_K(ID/OOD)={contr['id']['energy_norm_mean']:.3f}/{contr['ood_lame']['energy_norm_mean']:.3f}", flush=True)

    # Aggregate 5 seeds
    summary = {
        "train_time_single_s": _seed_stats([per_seed[str(s)]["train_time_single_s"] for s in SEEDS]),
        "train_time_ens5_s": _seed_stats([per_seed[str(s)]["train_time_ens5_s"] for s in SEEDS]),
        "splits": {},
        "energy_contraction": {},
        "timings": {},
    }
    for sp_k in ("id", "fine_paired", "ood_lame"):
        summary["energy_contraction"][sp_k] = {
            mk: _seed_stats([per_seed[str(s)]["energy_contraction"][sp_k][mk] for s in SEEDS])
            for mk in ("energy_norm_mean", "energy_norm_max", "frac_lt_1")
        }
        sp_sum = {"point_error": {}, "fidelity": {}, "displacement": {}, "von_mises": {}}
        for pk in per_seed[str(SEEDS[0])]["splits"][sp_k]["point_error"]:
            sp_sum["point_error"][pk] = _seed_stats([per_seed[str(s)]["splits"][sp_k]["point_error"][pk] for s in SEEDS])
        for fk, fdict in per_seed[str(SEEDS[0])]["splits"][sp_k]["fidelity"].items():
            sp_sum["fidelity"][fk] = {
                mk: _seed_stats([per_seed[str(s)]["splits"][sp_k]["fidelity"][fk][mk] for s in SEEDS])
                for mk in fdict
            }
        for kind, mlist in (("displacement", disp_methods), ("von_mises", vm_methods)):
            for mname in mlist:
                mkeys = list(per_seed[str(SEEDS[0])]["splits"][sp_k][kind][mname].keys())
                sp_sum[kind][mname] = {
                    mk: _seed_stats([per_seed[str(s)]["splits"][sp_k][kind][mname][mk] for s in SEEDS])
                    for mk in mkeys
                }
        summary["splits"][sp_k] = sp_sum

    for t_lbl in ("coarse", "fine"):
        summary["timings"][t_lbl] = {
            mk: _seed_stats([per_seed[str(s)]["timings"][t_lbl][mk] for s in SEEDS])
            for mk in per_seed[str(SEEDS[0])]["timings"][t_lbl]
        }

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t_all,
        "config": ELAST_CFG,
        "meshes": {
            "coarse": {k: mesh_c[k] for k in ("n_pts", "n_tri", "n_free_nodes", "n_free_dofs", "min_angle_deg", "mean_min_angle_deg")},
            "fine": {k: mesh_f[k] for k in ("n_pts", "n_tri", "n_free_nodes", "n_free_dofs", "min_angle_deg", "mean_min_angle_deg")},
        },
        "multiscale_amg": amg_scaling,
        "summary": summary,
        "per_seed": per_seed,
    }
    path = OUT / "task1_elasticity.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Rev4 Task 1] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.unit_only:
        mc = build_perforated_plate_mesh(0.0165, 0.037, n_edge=30, n_hole=76, smooth_iters=26, verify=True)
        mf = build_perforated_plate_mesh(0.0085, 0.019, n_edge=60, n_hole=150, smooth_iters=26, verify=True)
        print(f"T_c: {mc['n_pts']} nodes, {mc['n_tri']} tris, {mc['n_free_dofs']} DOFs, minAng={mc['min_angle_deg']:.2f} deg")
        print(f"T_f: {mf['n_pts']} nodes, {mf['n_tri']} tris, {mf['n_free_dofs']} DOFs, minAng={mf['min_angle_deg']:.2f} deg")
        unit_test_fem_and_vm(mc)
        unit_test_vector_amg(mc, ELAST_CFG)
        run_multiscale_amg_benchmark(ELAST_CFG)
    else:
        run_task1_elasticity(workers=args.workers)
