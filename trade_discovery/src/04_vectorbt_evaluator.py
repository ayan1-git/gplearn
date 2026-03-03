import vectorbt as vbt
import pandas as pd
import numpy as np
import importlib
try:
    config = importlib.import_module("src.config")
    DEFAULT_LONG_PCT = config.ENTRY_PCT / 100.0
    DEFAULT_SHORT_PCT = config.EXIT_PCT / 100.0
except (ImportError, AttributeError):
    DEFAULT_LONG_PCT, DEFAULT_SHORT_PCT = 0.80, 0.20

def evaluate_formula_with_vectorbt(
    gp_model, df_features_oos, df_raw_oos, train_signals_sorted,
    fees: float = 0.0003, slippage: float = 0.0001,
    long_pct: float = DEFAULT_LONG_PCT, short_pct: float = DEFAULT_SHORT_PCT
):
    """
    Evaluates the GP formula out-of-sample using VectorBT.
    
    Option B — Train-Calibrated Ranks:
    Each OOS signal is ranked against the TRAINING distribution.
    A signal must be in the top X% of training values to trigger a long,
    and bottom X% to trigger a short. If market regime changes, the formula
    fires fewer signals — which is the honest and correct behaviour.
    """
    print("Predicting signals on Out-of-Sample data...")
    raw_signals = gp_model.predict(df_features_oos.values)

    # ── OPTION B CORE: rank each OOS value within the training distribution ──
    # searchsorted finds where each OOS signal would sit in sorted train array.
    # Dividing by train length converts position → percentile rank (0.0 to 1.0).
    n_train = len(train_signals_sorted)
    oos_ranks_in_train = np.searchsorted(train_signals_sorted, raw_signals) / n_train

    long_entries  = oos_ranks_in_train > long_pct   # top 10% of training range
    short_entries = oos_ranks_in_train < short_pct  # bottom 10% of training range

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
