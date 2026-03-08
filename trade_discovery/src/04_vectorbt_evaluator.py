import importlib
import numpy as np
import pandas as pd
import vectorbt as vbt

pd.set_option("future.no_silent_downcasting", True)

try:
    config = importlib.import_module("src.config")
    DEFAULT_FEES = float(config.FEE_PER_SIDE)
    DEFAULT_SLIPPAGE = float(config.SLIPPAGE)
    DEFAULT_TP_MULT = float(config.TP_ATR_MULT)
    DEFAULT_SL_MULT = float(config.SL_ATR_MULT)
    DEFAULT_ABSOLUTE_EDGE_FLOOR = float(config.ABSOLUTE_EDGE_FLOOR)
except (ImportError, AttributeError):
    DEFAULT_FEES = 0.0003
    DEFAULT_SLIPPAGE = 0.0001
    DEFAULT_TP_MULT = 3.0
    DEFAULT_SL_MULT = 1.5
    DEFAULT_ABSOLUTE_EDGE_FLOOR = 0.0010


def evaluate_formula_with_vectorbt(
    gp_model,
    df_features_oos: pd.DataFrame,
    df_raw_oos: pd.DataFrame,
    long_pct_level: float,
    short_pct_level: float,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    rolling_window: int = 500,
    absolute_edge_floor: float = DEFAULT_ABSOLUTE_EDGE_FLOOR,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
):
    """
    Evaluate GP formula using asymmetric Triple Barrier Method.
    """
    if not 0.0 <= short_pct_level <= 100.0:
        raise ValueError("short_pct_level must be in [0, 100]")
    if not 0.0 <= long_pct_level <= 100.0:
        raise ValueError("long_pct_level must be in [0, 100]")
    if short_pct_level >= long_pct_level:
        raise ValueError("short_pct_level must be < long_pct_level")

    required_cols = {"open", "high", "low", "close"}
    missing_cols = required_cols - set(df_raw_oos.columns)
    if missing_cols:
        raise ValueError(f"df_raw_oos missing required columns: {sorted(missing_cols)}")

    print("Predicting signals on Out-of-Sample data...")
    raw_scores = gp_model.predict(df_features_oos.values)
    scores = pd.Series(raw_scores, index=df_features_oos.index, name="gp_score")

    # Causal rolling percentile rank against recent history only
    min_periods = min(50, rolling_window)
    rolling_ranks = scores.rolling(window=rolling_window, min_periods=min_periods).rank(pct=True)

    # Warmup fallback
    expanding_ranks = scores.expanding(min_periods=1).rank(pct=True)
    causal_ranks = rolling_ranks.fillna(expanding_ranks)

    long_rank_mask = causal_ranks >= (long_pct_level / 100.0)
    short_rank_mask = causal_ranks <= (short_pct_level / 100.0)

    # Absolute edge filter
    long_edge_mask = scores >= absolute_edge_floor
    short_edge_mask = scores <= -absolute_edge_floor

    long_entries = (long_rank_mask & long_edge_mask).astype(bool)
    short_entries = (short_rank_mask & short_edge_mask).astype(bool)

    n_long = int(long_entries.sum())
    n_short = int(short_entries.sum())
    coverage = ((n_long + n_short) / max(len(scores), 1)) * 100.0

    print(
        f"-> OOS Signal Coverage | Longs: {n_long} | Shorts: {n_short} | "
        f"Total: {coverage:.1f}% | TP/SL: {tp_mult}/{sl_mult}"
    )

    if n_long == 0 and n_short == 0:
        print("-> WARNING: No OOS signals passed filters.")

    entries_series = pd.Series(long_entries, index=df_features_oos.index)
    short_entries_series = pd.Series(short_entries, index=df_features_oos.index)

    # ATR Calculations
    high_low = df_raw_oos["high"] - df_raw_oos["low"]
    high_close = (df_raw_oos["high"] - df_raw_oos["close"].shift(1)).abs()
    low_close = (df_raw_oos["low"] - df_raw_oos["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_series = true_range.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    atr = atr_series.reindex(df_features_oos.index)

    close_prices = df_raw_oos.loc[df_features_oos.index, "close"]
    open_prices = df_raw_oos.loc[df_features_oos.index, "open"]

    # Trailing Stop (SL) and Take Profit (TP)
    sl_pct_series = (atr / close_prices) * sl_mult
    tp_pct_series = (atr / close_prices) * tp_mult

    sl_pct_series = sl_pct_series.ffill().fillna(0.01).clip(lower=0.001)
    tp_pct_series = tp_pct_series.ffill().fillna(0.01).clip(lower=0.001)

    # Execute on next bar open
    entries_shifted = entries_series.shift(1).fillna(False).astype(bool)
    short_entries_shifted = short_entries_series.shift(1).fillna(False).astype(bool)
    
    # Exits only when opposite signal triggers (Cross-Close)
    exits_shifted = short_entries_shifted
    short_exits_shifted = entries_shifted

    print("Running VectorBT backtest (Asymmetric TBM)...")
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        price=open_prices,
        entries=entries_shifted,
        exits=exits_shifted,
        short_entries=short_entries_shifted,
        short_exits=short_exits_shifted,
        fees=fees,
        slippage=slippage,
        sl_stop=sl_pct_series,
        tp_stop=tp_pct_series,
        sl_trail=True,
        freq="30min"
    )

    stats = portfolio.stats()
    print("\n--- Out-of-Sample Results ---")
    print(stats[["Total Return [%]", "Max Drawdown [%]", "Win Rate [%]", "Sharpe Ratio"]])

    metadata = {
        "n_long": n_long,
        "n_short": n_short,
        "coverage_pct": float(coverage),
        "tp_mult": float(tp_mult),
        "sl_mult": float(sl_mult),
    }

    return portfolio, stats, metadata
