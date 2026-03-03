import vectorbt as vbt
import pandas as pd
import numpy as np

def evaluate_formula_with_vectorbt(
    gp_model, 
    df_features_oos, 
    df_raw_oos, 
    entry_threshold: float,
    exit_threshold: float,
    fees: float = 0.0003, 
    slippage: float = 0.0001
):
    """
    Evaluates the GP formula out-of-sample using VectorBT.
    
    Uses absolute thresholds calculated strictly from the training distribution 
    to prevent data leakage and guarantee consistency between backtest and live execution.
    """
    print("Predicting signals on Out-of-Sample data...")
    raw_signals = gp_model.predict(df_features_oos.values)

    # ── FIXED: Direct Absolute Thresholding ──
    # Uses the exact cutoff values computed from the training fold pipeline.
    long_entries  = raw_signals > entry_threshold
    short_entries = raw_signals < exit_threshold

    n_long  = long_entries.sum()
    n_short = short_entries.sum()
    coverage = (n_long + n_short) / len(raw_signals) * 100
    print(f"-> OOS Signal Coverage | Longs: {n_long} | Shorts: {n_short} | Total: {coverage:.1f}% of bars")

    if n_long == 0 and n_short == 0:
        print("-> WARNING: Formula produced ZERO signals in OOS. Regime likely changed. Fold will fail survival gate.")

    entries_series       = pd.Series(long_entries,  index=df_features_oos.index)
    short_entries_series = pd.Series(short_entries, index=df_features_oos.index)

    # Reversal logic: long entry closes short, short entry closes long
    exits_series       = short_entries_series
    short_exits_series = entries_series

    # ── ATR TRAILING STOP (Wilder RMA) ──
    high_low   = df_raw_oos['high'] - df_raw_oos['low']
    high_close = np.abs(df_raw_oos['high'] - df_raw_oos['close'].shift(1))
    low_close  = np.abs(df_raw_oos['low']  - df_raw_oos['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_series = true_range.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean()
    atr        = atr_series.reindex(df_features_oos.index)

    close_prices = df_raw_oos.loc[df_features_oos.index, 'close']

    from src.config import ORACLE_ATR_MULT          # single source of truth
    atr_pct_raw    = (atr / close_prices) * ORACLE_ATR_MULT
    atr_pct_series = atr_pct_raw.ffill().fillna(0.01).clip(lower=0.001)

    print("Running VectorBT Backtest with Long/Short and Dynamic ATR Trailing Stop...")
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries_series,
        exits=exits_series,
        short_entries=short_entries_series,
        short_exits=short_exits_series,
        fees=fees,
        slippage=slippage,
        sl_stop=atr_pct_series,
        sl_trail=True,
        freq='30min'
    )

    stats = portfolio.stats()
    print("\n--- Out-of-Sample Results ---")
    print(stats[['Total Return [%]', 'Max Drawdown [%]', 'Win Rate [%]', 'Sharpe Ratio']])

    metadata = {
        'n_long': int(n_long),
        'n_short': int(n_short),
        'coverage_pct': float(coverage)
    }

    return portfolio, stats, metadata
