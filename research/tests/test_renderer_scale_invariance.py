"""修訂 2 的回歸測試: render() 對絕對波動尺度不敏感(同一路徑放大 σ,圖片幾乎不變)。"""
import numpy as np

from research.pipeline.render_market import render


def _window(sigma, seed):
    r = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(sigma * r.standard_normal(48)))
    open_ = np.r_[30000, close[:-1]]
    hi = np.maximum(open_, close) * (1 + sigma * 0.3 * r.random(48))
    lo = np.minimum(open_, close) * (1 - sigma * 0.3 * r.random(48))
    return np.column_stack([open_, hi, lo, close, np.exp(r.normal(3, 0.3, 48))])


def test_volatility_scale_is_invisible_in_render():
    same_path = np.mean([np.abs(render(_window(0.0005, s)).astype(float) - render(_window(0.004, s)).astype(float)).mean() for s in range(20)])
    other_path = np.mean([np.abs(render(_window(0.0015, s)).astype(float) - render(_window(0.0015, s + 100)).astype(float)).mean() for s in range(20)])
    assert same_path < 1.0 < 10.0 < other_path
