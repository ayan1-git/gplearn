import vectorbt as vbt
import pandas as pd
import numpy as np

def evaluate_formula_with_vectorbt(
    gp_model, 
    df_features_oos, 
    df_raw_oos, 
    long_pct_threshold: float,
    short_pct_threshold: float,
    fees: float = 0.0003, 
    slippage: float = 0.0001,
    rolling_window: int = 500  # e.g., ~2 weeks of 30min bars
):
    """
    Evaluates GP formula using a CAUSAL rolling rank.
    Prevents both Distribution Shift (by adapting to recent regime)
    and Lookahead Bias (by only ranking against past bars, not future OOS bars).
    """
    print("Predicting signals on Out-of-Sample data...")
    raw_signals = gp_model.predict(df_features_oos.values)
    signals_series = pd.Series(raw_signals, index=df_features_oos.index)

    # ── THE FIX: Causal Rolling Percentile ──
    # Rank today's signal against the LAST `rolling_window` bars only.
    # pct=True returns a value between 0.0 and 1.0
    rolling_ranks = signals_series.rolling(window=rolling_window, min_periods=50).rank(pct=True)
    
    # Fill the initial warm-up period by ranking against whatever we have so far (expanding)
    expanding_ranks = signals_series.expanding(min_periods=1).rank(pct=True)
    causal_ranks = rolling_ranks.fillna(expanding_ranks)

    # Trade if the current bar is in the top/bottom X% of RECENT history
    long_entries  = causal_ranks > (long_pct_threshold / 100.0)    # e.g. > 0.80
    short_entries = causal_ranks < (short_pct_threshold / 100.0)   # e.g. < 0.20

    n_long  = long_entries.sum()
    n_short = short_entries.sum()
    coverage = (n_long + n_short) / len(raw_signals) * 100
    print(f"-> OOS Signal Coverage | Longs: {n_long} | Shorts: {n_short} | Total: {coverage:.1f}% of bars")

    if n_long == 0 and n_short == 0:
        print("-> WARNING: Formula produced ZERO signals in OOS. Regime likely changed.")

    entries_series       = pd.Series(long_entries,  index=df_features_oos.index)
    short_entries_series = pd.Series(short_entries, index=df_features_oos.index)
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
