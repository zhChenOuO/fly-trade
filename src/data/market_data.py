"""加密貨幣 OHLCV 資料來源。

用 ccxt 抓交易所的公開歷史 K 線資料。這些都是「市場行情」端點,任何
人都能查、不需要註冊帳號也不需要 API key/secret(API key 是下單、查
自己帳戶餘額才需要的東西)。

雲端沙盒環境常常連不到 Binance(要嘛是網路白名單擋掉,要嘛是 Binance
本身會擋一些雲端服務商的 IP 範圍),但在你自己家裡/辦公室的網路
(包含台灣一般的住宅網路)通常是直接可以連的。保險起見這裡預設會
依序嘗試好幾個交易所,任何一個能連就用那個,全部都連不到才會退回合
成資料。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 依序嘗試的交易所,全部都是公開 OHLCV 端點、不需要 API key。
# 如果你所在地區某個交易所被擋(例如企業網路擋 Binance),它就會自動
# 跳到下一個。也可以在 configs/*.yaml 裡自訂這個順序。
DEFAULT_EXCHANGE_FALLBACKS = ["binance", "okx", "kraken", "coinbase", "bybit"]


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 2000,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    import ccxt

    exchange = getattr(ccxt, exchange_id)()
    # 交易所單次請求有上限(binance 1000 根),要用 since 分頁往後抓。
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - limit * tf_ms
    raw = []
    while len(raw) < limit:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not chunk:
            break
        raw += chunk
        since = chunk[-1][0] + tf_ms
        if len(chunk) < 2:
            break
    raw = raw[-limit:]
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df


def fetch_ohlcv_any(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 2000,
    exchange_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """依序嘗試 exchange_ids 裡的每個交易所,回傳第一個成功的 (df, exchange_id)。

    注意:不同交易所的交易對命名可能略有差異(例如 coinbase 用
    "BTC/USD" 而不是 "BTC/USDT"),如果某個交易所因為找不到交易對而
    失敗,也會被當成失敗、自動換下一個。
    """
    exchange_ids = exchange_ids or DEFAULT_EXCHANGE_FALLBACKS
    errors = {}
    for eid in exchange_ids:
        try:
            df = fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, limit=limit, exchange_id=eid)
            return df, eid
        except Exception as e:
            errors[eid] = repr(e)
    detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"所有交易所都抓不到資料。逐一嘗試的錯誤: {detail}")


def make_synthetic_ohlcv(
    n_bars: int = 2000,
    start_price: float = 30000.0,
    mu: float = 0.0,
    sigma: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """幾何布朗運動生成的假 K 線,只用於離線測試 pipeline 是否能跑通,
    不代表真實市場行為,回測結果沒有任何實際交易意義。
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, size=n_bars)
    close = start_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, sigma / 2, size=n_bars)))
    low = close * (1 - np.abs(rng.normal(0, sigma / 2, size=n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.lognormal(mean=5, sigma=1, size=n_bars)

    idx = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def load_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 2000,
    use_synthetic_fallback: bool = True,
    exchange_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """依序嘗試多個公開交易所,全部失敗才退回合成資料。回傳 (df, source)。"""
    try:
        df, eid = fetch_ohlcv_any(symbol=symbol, timeframe=timeframe, limit=limit, exchange_ids=exchange_ids)
        return df, eid
    except Exception as e:
        if not use_synthetic_fallback:
            raise
        print(f"[market_data] 所有交易所都抓不到真實資料 ({e!r}),改用合成資料。")
        return make_synthetic_ohlcv(n_bars=limit), "synthetic"


def load_ohlcv_cached(symbol: str, timeframe: str, limit: int, exchange_ids: list[str] | None = None,
                      cache_dir: str = "data_cache") -> pd.DataFrame:
    """第一次抓完存 CSV,之後所有人(含各個調參 agent)讀同一份,結果才能互相比較。"""
    from pathlib import Path
    path = Path(cache_dir) / f"{symbol.replace('/', '_')}_{timeframe}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df, _ = fetch_ohlcv_any(symbol, timeframe, limit, exchange_ids)
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path)
    return df
