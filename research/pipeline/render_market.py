"""Deterministic, text-free candles and volume; only the supplied 48 bars exist."""
import numpy as np


def render(window: np.ndarray) -> np.ndarray:
    """Render OHLCV (48, 5) to uint8 RGB (64, 64, 3), normalized per window.

    Candles occupy x=8..55, y=2..47; volume occupies y=52..61.
    The remaining pixels are black. No timestamps, axes, text, or metadata.
    """
    bars = np.asarray(window, dtype=np.float64)
    if bars.shape != (48, 5) or not np.isfinite(bars).all():
        raise ValueError("window must contain finite OHLCV values with shape (48, 5)")
    opening, high, low, close, volume = bars.T
    if (np.any(bars[:, :4] <= 0) or np.any(volume < 0)
            or np.any(low > np.minimum(opening, close))
            or np.any(high < np.maximum(opening, close))):
        raise ValueError("invalid OHLCV bounds")

    bottom, top = low.min(), high.max()
    if top == bottom:
        price_y = np.full((48, 4), 25, dtype=int)
    else:
        price_y = np.rint(2 + (top - bars[:, :4]) / (top - bottom) * 45).astype(int)
    volume_height = np.rint(volume / volume.max() * 10).astype(int) if volume.max() else np.zeros(48, dtype=int)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    for i, (oy, hy, ly, cy) in enumerate(price_y):
        color = (46, 204, 113) if close[i] >= opening[i] else (231, 76, 60)
        x = 8 + i
        image[hy:ly + 1, x] = tuple(c // 2 for c in color)
        image[min(oy, cy):max(oy, cy) + 1, x] = color
        if volume_height[i]:
            image[62 - volume_height[i]:62, x] = color
    return image
