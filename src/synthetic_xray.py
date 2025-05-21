"""
synthetic_xray.py
---------------------------------
Utility functions for generating synthetic X-ray phantom images.
The algorithm is intentionally lightweight so it runs in real time.
"""

from __future__ import annotations

from typing import Tuple
import math
import cv2
import numpy as np

def generate_phantom(
    size: int = 512,
    n_bones: int = 8,
    bone_min_radius: float = 0.04,
    bone_max_radius: float = 0.12,
    noise_sigma: float = 10.0,
) -> np.ndarray:
    """
    Return a uint8 grayscale phantom resembling a plain-film X-ray.

    Parameters
    ----------
    size : int
        Image width/height (square).
    n_bones : int
        How many circular "bones" to embed.
    bone_min_radius, bone_max_radius : float
        Radius range (fraction of `size`).
    noise_sigma : float
        \sigma of Gaussian noise to mimic detector noise.
    
    Notes
    -----
    This is *not* physically accurate. Replace with your
    Monte-Carlo / CT-DRR pipeline if you need realism.
    """
    rng = np.random.default_rng()

    # 1) soft-tissue baseline (high ≈ radiolucent → bright on negative)
    img = rng.normal(loc=220.0, scale=noise_sigma, size=(size, size)).astype(np.float32)

    # 2) bones (darker shadows -> lower intensity)
    for _ in range(n_bones):
        center = rng.integers(low=int(size * 0.1), high=int(size * 0.9), size=2)
        radius = rng.integers(
            low=int(size * bone_min_radius), high=int(size * bone_max_radius)
        )
        cv2.circle(img, tuple(center), int(radius), color=int(40 + rng.integers(100)), thickness=-1)
    
    # 2-b) catheter & guidewire
    img = _draw_curve(
        img, 
        thickness = rng.integers(3,6),
        intensity = rng.integers(30, 60),
    )
    img = _draw_curve(
        img, 
        thickness = 1,
        intensity = rng.integers(20, 40),
    )

    # 3) soft global blutr to mimic scatter
    cv2.GaussianBlur(img, (0, 0), sigmaX=size * 0.01, dst=img)

    # 4) clamp -> uint8
    return np.clip(img, 0, 255).astype(np.uint8)

def _draw_curve(
        canvas: np.ndarray,
        thickness: int,
        intensity: int,
) -> np.ndarray:
    """
    draw catheter/guidewire with Bezier curve
    `intensity` is 0=black, 255=white
    """
    h, w = canvas.shape
    rng = np.random.default_rng()
    # 3~5 random generated control points
    n_pts = rng.integers(3, 6)
    points = rng.integers(low=[w//8, h//8], high=[w*7//8, h*7//8], size=(n_pts, 2))
    # Bezier sampling
    t = np.linspace(0, 1, num=500)
    curve = _bezier(points, t)
    for (x, y) in curve.astype(int):
        cv2.circle(canvas, (int(x), int(y)), thickness, int(intensity), -1)
    return canvas

def _bezier(ctrl_pts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ evaluate Bezier curve numpy vectorization """
    n = len(ctrl_pts) - 1
    coeff = [
        math.comb(n, k) * ((1 - t) ** (n - k)) * (t ** k) for k in range(n + 1)
    ]
    return np.tensordot(np.array(coeff).T, ctrl_pts, axes=1)