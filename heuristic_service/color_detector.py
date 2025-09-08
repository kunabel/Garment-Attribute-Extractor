import cv2
import numpy as np
from typing import Dict, Any


def _grabcut_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Create a foreground mask using GrabCut with a loose center rectangle.
    Returns a uint8 mask with 1 for foreground, 0 for background.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Loose rectangle inside the image (5% margin)
    x0 = int(0.05 * w)
    y0 = int(0.05 * h)
    rect = (x0, y0, w - 2 * x0, h - 2 * y0)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        # Fallback: assume center region is garment
        fg = np.zeros((h, w), np.uint8)
        cx0 = int(0.2 * w); cy0 = int(0.2 * h)
        cx1 = int(0.8 * w); cy1 = int(0.8 * h)
        fg[cy0:cy1, cx0:cx1] = 1
    return fg


def _circular_mean_hue(h_degrees: np.ndarray) -> float:
    """
    Compute circular mean of hue angles in degrees [0,360).
    """
    radians = np.deg2rad(h_degrees)
    s = np.sin(radians).mean()
    c = np.cos(radians).mean()
    angle = np.rad2deg(np.arctan2(s, c)) % 360.0
    return float(angle)


def _name_from_hsv_stats(h_mean: float, s_mean: float, v_mean: float) -> str:
    """
    Map average HSV to a coarse, human-friendly color name.
    HSV ranges use OpenCV scale: H in [0,180], S,V in [0,255].
    We'll convert H to [0,360) degrees first.
    """
    h_deg = (h_mean * 2.0) % 360.0  # OpenCV hue is 0..180
    s = s_mean / 255.0
    v = v_mean / 255.0

    # Achromatic buckets
    if s < 0.12:  # very low saturation
        if v < 0.25:
            return "black"
        elif v < 0.75:
            return "gray"
        else:
            return "white"

    # Brown heuristic: orange-ish hue + darker value + moderate saturation
    if 10 <= h_deg < 45 and v < 0.65 and s > 0.25:
        return "brown"

    # Chromatic ranges (simple coarse bins)
    if h_deg >= 345 or h_deg < 15:
        return "red"
    if 15 <= h_deg < 30:
        return "orange"
    if 30 <= h_deg < 55:
        return "yellow"
    if 55 <= h_deg < 85:
        return "lime"
    if 85 <= h_deg < 150:
        return "green"
    if 150 <= h_deg < 190:
        return "cyan"
    if 190 <= h_deg < 250:
        return "blue"
    if 250 <= h_deg < 290:
        return "purple"
    if 290 <= h_deg < 330:
        return "magenta"
    if 330 <= h_deg < 345:
        return "pink"

    return "unknown"


def detect_garment_color(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Detect dominant garment color from a BGR image (OpenCV).
    Returns a dict with a coarse color name and representative RGB tuple.

    Steps:
      1) GrabCut foreground mask (background-agnostic).
      2) Extract garment pixels and compute HSV stats.
      3) Name color via simple HSV rules (no webcolors).
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image provided")

    # Optional downscale for speed
    h, w = img_bgr.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr

    # Foreground mask
    fg_mask = _grabcut_mask(img_small)
    fg_pixels = img_small[fg_mask == 1]
    if fg_pixels.size == 0:
        # Fallback: use whole image if segmentation failed
        fg_pixels = img_small.reshape(-1, 3)

    # Convert to HSV
    hsv = cv2.cvtColor(fg_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    H = hsv[:, 0].astype(np.float32)   # 0..180
    S = hsv[:, 1].astype(np.float32)   # 0..255
    V = hsv[:, 2].astype(np.float32)   # 0..255

    # If many pixels are nearly white shadows or near-black, keep only "valid" ones first
    valid = (V > 20)  # drop very dark noise
    if valid.any():
        H, S, V = H[valid], S[valid], V[valid]

    # Compute central tendency
    # For Hue use circular mean on degrees
    h_mean_deg = _circular_mean_hue(H * 2.0) if H.size else 0.0
    s_mean = float(S.mean() if S.size else 0.0)
    v_mean = float(V.mean() if V.size else 0.0)

    # Name color
    color_name = _name_from_hsv_stats(h_mean=h_mean_deg / 2.0,  # convert back to OpenCV 0..180 for mapping fn
                                      s_mean=s_mean,
                                      v_mean=v_mean)

    # Representative RGB (average of garment pixels)
    mean_bgr = fg_pixels.mean(axis=0).astype(np.uint8).tolist()  # [B,G,R]
    rep_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))

    return {
        "color_name": color_name,
        "rgb": rep_rgb,
        "hsv_mean": (round(h_mean_deg, 1), round(s_mean, 1), round(v_mean, 1)),
    }
