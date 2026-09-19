"""Retina (compound eye) PCA projection and encoder for Drosophila photoreceptors.

Contract (from research/SPEC_v3.md §11):
- Independent PCA on 3D coordinates (pos_x, pos_y, pos_z) of R1-6 for each eye (left & right).
- First 2 principal axes standardized to [0, 1]^2 to obtain (u, v).
- Fixed axis sign convention: 1st axis positively correlated with pos_x,
  2nd axis positively correlated with pos_y.
- Both eyes view the full 64x64 image (u -> image x, v -> image y).
- R7 and R8 take (u, v) from their 3D nearest-neighbor R1-6 of the same eye.
- Bilinear sampling from normalized image (x - train_mean) / 128.0:
  - R1-6: luminance (mean RGB)
  - R7: R channel (channel 0)
  - R8: G channel (channel 1)
- input_scale = 1.0.
- Output shape: (B, n_sensory) currents.
- Mapping depends strictly on photoreceptor coordinates and train mean; no labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.connectome.schema import ConnectomeGraph


def compute_retina_coordinates(
    pr_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute (u, v) retina coordinates for photoreceptors according to SPEC_v3.

    Parameters
    ----------
    pr_df: pd.DataFrame
        DataFrame containing photoreceptor rows, must include columns:
        'cell_type', 'side', 'pos_x', 'pos_y', 'pos_z'.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'u': (N,) array of float32 in [0, 1]
        - 'v': (N,) array of float32 in [0, 1]
        - 'uv': (N, 2) array of float32 in [0, 1]^2
        - 'photoreceptor_type': (N,) array of str ('R1-6', 'R7', 'R8')
        - 'eye': (N,) array of str ('left', 'right')
    """
    n_pr = len(pr_df)
    u_all = np.zeros(n_pr, dtype=np.float32)
    v_all = np.zeros(n_pr, dtype=np.float32)

    cell_types = pr_df["cell_type"].values.astype(str)
    sides = pr_df["side"].values.astype(str)

    pos_x = pr_df["pos_x"].values.astype(np.float64)
    pos_y = pr_df["pos_y"].values.astype(np.float64)
    pos_z = pr_df["pos_z"].values.astype(np.float64)

    for side in ["left", "right"]:
        eye_mask = sides == side
        r16_mask = eye_mask & (cell_types == "R1-6")
        r16_indices = np.where(r16_mask)[0]

        if len(r16_indices) < 3:
            raise ValueError(f"Too few R1-6 photoreceptors for eye '{side}': {len(r16_indices)}")

        # 1. 3D coordinates of R1-6
        xyz_r16 = np.column_stack([pos_x[r16_indices], pos_y[r16_indices], pos_z[r16_indices]])
        mean_xyz = xyz_r16.mean(axis=0)
        xyz_c = xyz_r16 - mean_xyz

        # 2. PCA via SVD
        _, _, vt = np.linalg.svd(xyz_c, full_matrices=False)
        v1 = vt[0].copy()
        v2 = vt[1].copy()

        z1 = xyz_c @ v1
        z2 = xyz_c @ v2

        # 3. Fixed axis sign convention:
        # 1st axis positively correlated with pos_x
        corr1 = np.corrcoef(z1, xyz_r16[:, 0])[0, 1]
        if np.isnan(corr1) or corr1 < 0:
            v1 = -v1
            z1 = -z1

        # 2nd axis positively correlated with pos_y
        corr2 = np.corrcoef(z2, xyz_r16[:, 1])[0, 1]
        if np.isnan(corr2) or corr2 < 0:
            v2 = -v2
            z2 = -z2

        # 4. Standardize to [0, 1]^2
        range_z1 = z1.max() - z1.min()
        range_z2 = z2.max() - z2.min()
        u_r16 = (z1 - z1.min()) / (range_z1 if range_z1 > 1e-9 else 1.0)
        v_r16 = (z2 - z2.min()) / (range_z2 if range_z2 > 1e-9 else 1.0)

        u_all[r16_indices] = u_r16.astype(np.float32)
        v_all[r16_indices] = v_r16.astype(np.float32)

        # 5. R7 and R8 take (u, v) from 3D nearest-neighbor R1-6 of the same eye
        r78_mask = eye_mask & ((cell_types == "R7") | (cell_types == "R8"))
        r78_indices = np.where(r78_mask)[0]

        if len(r78_indices) > 0:
            xyz_r78 = np.column_stack([pos_x[r78_indices], pos_y[r78_indices], pos_z[r78_indices]])
            tree = cKDTree(xyz_r16)
            _, nn_idx = tree.query(xyz_r78)
            u_all[r78_indices] = u_r16[nn_idx].astype(np.float32)
            v_all[r78_indices] = v_r16[nn_idx].astype(np.float32)

    # Final check: clip to [0, 1] and verify no NaNs
    u_all = np.clip(u_all, 0.0, 1.0)
    v_all = np.clip(v_all, 0.0, 1.0)

    return {
        "u": u_all,
        "v": v_all,
        "uv": np.column_stack([u_all, v_all]),
        "photoreceptor_type": cell_types,
        "eye": sides,
    }


def bilinear_sample_retina(
    norm_images: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    photoreceptor_type: np.ndarray | list[str],
) -> np.ndarray:
    """Sample pixel channels at (u, v) using bilinear interpolation.

    Parameters
    ----------
    norm_images: np.ndarray
        Pre-normalized image batch of shape (B, 64, 64, 3).
    u: np.ndarray
        Horizontal coordinates in [0, 1] of shape (n_sensory,).
    v: np.ndarray
        Vertical coordinates in [0, 1] of shape (n_sensory,).
    photoreceptor_type: array-like of str
        Cell type for each sensory neuron ('R1-6', 'R7', 'R8').

    Returns
    -------
    np.ndarray
        Sampled sensory currents of shape (B, n_sensory), dtype float32.
    """
    B, H, W, C = norm_images.shape
    n_sensory = len(u)

    # Map [0, 1] to continuous grid indices [0, W - 1] and [0, H - 1]
    x = u * (W - 1)
    y = v * (H - 1)

    x0 = np.floor(x).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    out = np.zeros((B, n_sensory), dtype=np.float32)
    pr_types = np.asarray(photoreceptor_type)

    # 1. R1-6: Luminance (mean RGB across color channels)
    mask_r16 = pr_types == "R1-6"
    if np.any(mask_r16):
        ch_lum = norm_images.mean(axis=-1)  # (B, H, W)
        p00 = ch_lum[:, y0[mask_r16], x0[mask_r16]]
        p10 = ch_lum[:, y0[mask_r16], x1[mask_r16]]
        p01 = ch_lum[:, y1[mask_r16], x0[mask_r16]]
        p11 = ch_lum[:, y1[mask_r16], x1[mask_r16]]
        out[:, mask_r16] = (
            p00 * w00[mask_r16]
            + p10 * w10[mask_r16]
            + p01 * w01[mask_r16]
            + p11 * w11[mask_r16]
        )

    # 2. R7: Red channel (channel 0)
    mask_r7 = pr_types == "R7"
    if np.any(mask_r7):
        ch_r = norm_images[:, :, :, 0]
        p00 = ch_r[:, y0[mask_r7], x0[mask_r7]]
        p10 = ch_r[:, y0[mask_r7], x1[mask_r7]]
        p01 = ch_r[:, y1[mask_r7], x0[mask_r7]]
        p11 = ch_r[:, y1[mask_r7], x1[mask_r7]]
        out[:, mask_r7] = (
            p00 * w00[mask_r7]
            + p10 * w10[mask_r7]
            + p01 * w01[mask_r7]
            + p11 * w11[mask_r7]
        )

    # 3. R8: Green channel (channel 1)
    mask_r8 = pr_types == "R8"
    if np.any(mask_r8):
        ch_g = norm_images[:, :, :, 1]
        p00 = ch_g[:, y0[mask_r8], x0[mask_r8]]
        p10 = ch_g[:, y0[mask_r8], x1[mask_r8]]
        p01 = ch_g[:, y1[mask_r8], x0[mask_r8]]
        p11 = ch_g[:, y1[mask_r8], x1[mask_r8]]
        out[:, mask_r8] = (
            p00 * w00[mask_r8]
            + p10 * w10[mask_r8]
            + p01 * w01[mask_r8]
            + p11 * w11[mask_r8]
        )

    return out


def retina_encoder(
    graph: ConnectomeGraph,
    mean_img: np.ndarray,
    input_scale: float = 1.0,
):
    """Create a retina encoder function mapping market images to sensory neuron currents.

    Parameters
    ----------
    graph: ConnectomeGraph
        Connectome graph containing sensory (u, v) and photoreceptor_type metadata.
    mean_img: np.ndarray
        Train mean image of shape (64, 64, 3) used for mean subtraction.
    input_scale: float
        Scaling factor for currents. Fixed to 1.0 in SPEC_v3.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        encode(images: uint8 (B, 64, 64, 3)) -> (B, n_sensory) float32 currents.
    """
    if "u" not in graph.meta or "v" not in graph.meta:
        raise ValueError("graph.meta must contain 'u' and 'v' coordinates for retina encoding")

    u = np.asarray(graph.meta["u"], dtype=np.float32)
    v = np.asarray(graph.meta["v"], dtype=np.float32)
    pr_type = np.asarray(graph.meta.get("photoreceptor_type", ["R1-6"] * len(u)))

    mean_f32 = np.asarray(mean_img, dtype=np.float32)
    if mean_f32.shape != (64, 64, 3):
        raise ValueError(f"Expected mean_img shape (64, 64, 3), got {mean_f32.shape}")

    def encode(images: np.ndarray) -> np.ndarray:
        imgs = np.asarray(images)
        if imgs.ndim == 3:
            imgs = imgs[None, ...]
        if imgs.shape[1:] != (64, 64, 3):
            raise ValueError(f"Expected image shape (*, 64, 64, 3), got {imgs.shape}")

        # mean_sub encoding: (x - train_mean) / 128.0
        norm_img = (imgs.astype(np.float32) - mean_f32) / 128.0
        currents = bilinear_sample_retina(norm_img, u, v, pr_type) * float(input_scale)
        return currents

    return encode
