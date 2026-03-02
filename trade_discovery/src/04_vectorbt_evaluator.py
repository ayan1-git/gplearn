import vectorbt as vbt
import pandas as pd
import numpy as np

def evaluate_formula_with_vectorbt(
    gp_model, df_features_oos, df_raw_oos, entry_pct, exit_pct,
    fees: float = 0.0003, slippage: float = 0.0001
):
    """
    Evaluates the GP formula out-of-sample using VectorBT.
    Now correctly handles Long/Short symmetry and ATR-based dynamic trailing stops.
    """
    print("Predicting signals on Out-of-Sample data...")
    
    # Get the raw continuous signal from the math formula
    raw_signals = gp_model.predict(df_features_oos.values)

    # Use strict training-set percentiles to avoid future data leakage
    long_entries = raw_signals > entry_pct
    short_entries = raw_signals < exit_pct

    # Align arrays with the original timeframe
    entries_series = pd.Series(long_entries, index=df_features_oos.index)
    short_entries_series = pd.Series(short_entries, index=df_features_oos.index)
    
    # Reversal Logic: A Long entry closes a Short, and vice versa.
    # This perfectly mimics the Oracle's symmetric path-dependent scoring.
    exits_series = short_entries_series
    short_exits_series = entries_series

    # Wilder's ATR Calculation (RMA) to match Oracle parity exactly
    high_low   = df_raw_oos['high'] - df_raw_oos['low']
    high_close = np.abs(df_raw_oos['high'] - df_raw_oos['close'].shift(1))
    low_close  = np.abs(df_raw_oos['low']  - df_raw_oos['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Use alpha=1/period, adjust=False to match TradingView/Oracle RMA smoothing
    atr_series = true_range.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean()
    atr = atr_series.reindex(df_features_oos.index)
    
    close_prices = df_raw_oos.loc[df_features_oos.index, 'close']
    
    # Calculate raw ATR% - using 2.0 to match the current pipeline config
    atr_pct_raw = (atr / close_prices) * 3.5
    
    # A) Forward fill: Use the most recent valid ATR for subsequent NaNs
    atr_pct_series = atr_pct_raw.ffill()
    
    # B) Handle leading NaNs (start of dataset) with a safe 1% default
    atr_pct_series = atr_pct_series.fillna(0.01)
    
    # Clip to sanity bounds (min 0.1% stop)
    atr_pct_series = atr_pct_series.clip(lower=0.001)

    print("Running VectorBT Backtest with Long/Short and Dynamic ATR Trailing Stop...")

    # VectorBT portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries_series,              # Open Long
        exits=exits_series,                  # Close Long (when Short triggers)
        short_entries=short_entries_series,  # Open Short
        short_exits=short_exits_series,      # Close Short (when Long triggers)
        fees=fees,                           # Realistic transaction fee
        slippage=slippage,                   # Slippage
        sl_stop=atr_pct_series,              # Dynamic ATR% trailing stop
        sl_trail=True,                       # Trailing behavior enabled
        freq='30min'
    )

    stats = portfolio.stats()
    print("\n--- Out-of-Sample Results ---")
    print(stats[['Total Return [%]', 'Max Drawdown [%]', 'Win Rate [%]', 'Sharpe Ratio']])
    
    return portfolio, stats
