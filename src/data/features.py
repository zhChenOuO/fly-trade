"""從 OHLCV 產生要餵給感覺神經元的特徵向量。

特徵盡量保持在合理範圍(標準化過),因為這些數字會直接變成注入神經
元的電流,數值過大會讓 tanh 飽和、整個 reservoir 失去動態範圍。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> np.ndarray:
    """回傳標準化後的特徵矩陣 (T, n_features)。

    特徵包含:對數報酬率(多個時間尺度)、波動度、RSI、量能變化。
    """
    close = df["close"]
    log_ret_1 = np.log(close).diff(1)
    log_ret_4 = np.log(close).diff(4)
    log_ret_12 = np.log(close).diff(12)
    volatility = log_ret_1.rolling(12).std()
    rsi = _rsi(close) / 100.0 - 0.5  # 置中到 [-0.5, 0.5]
    vol_change = np.log(df["volume"].replace(0, np.nan)).diff(1)

    feat_df = pd.concat(
        {
            "ret_1": log_ret_1,
            "ret_4": log_ret_4,
            "ret_12": log_ret_12,
            "volatility": volatility,
            "rsi": rsi,
            "vol_change": vol_change,
        },
        axis=1,
    )
    # 因果 z-score:只用「過去 500 根」的均值/標準差,不用整段資料的統計量
    # (整段統計量會把未來資訊漏進特徵,也讓 train/val/test 切分不乾淨)。
    roll = feat_df.rolling(500, min_periods=50)
    feat_df = ((feat_df - roll.mean()) / (roll.std() + 1e-8)).fillna(0.0)
    feat = np.clip(feat_df.values, -4.0, 4.0)
    return feat


FEATURE_NAMES = ["ret_1", "ret_4", "ret_12", "volatility", "rsi", "vol_change"]
