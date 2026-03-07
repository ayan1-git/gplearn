import numpy as np
import numba
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE UTILITY: Wilder's ATR (RMA)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wilder_atr(df_raw: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes ATR using Wilder's smoothing (RMA).
    Maintains float64 precision for difference calculations to avoid 
    catastrophic cancellation on large absolute prices.
    """
    high_low   = df_raw['high'] - df_raw['low']
    high_close = np.abs(df_raw['high'] - df_raw['close'].shift(1))
    low_close  = np.abs(df_raw['low']  - df_raw['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder's smoothing: alpha = 1/period, no look-ahead (adjust=False)
    atr = true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return atr


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA CORE - STRICT FLOAT32 SIGNATURE
# 'f4' = float32, 'i8' = int64. This prevents Numba from silently compiling 
# a 64-bit version and blowing up your RAM.
# ─────────────────────────────────────────────────────────────────────────────

@numba.jit('f4[:](f4[:], f4[:], f4[:], f4[:], f4[:], i8, f4, f4, f4, f4, f4)', 
           nopython=True, cache=True, fastmath=True)
def run_oracle_scoring(
    open_arr, high_arr, low_arr, close_arr, atr_arr,
    max_hold,
    fee_per_side,
    slippage,
    atr_mult,
    saturation_factor,
    mae_penalty
):
    """
    Oracle 4.0: Risk-Standardized & Path-Dependent Targets.
    """
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    total_cost_pct = np.float32((fee_per_side + slippage) * 2.0)
    stop_distances = atr_arr * atr_mult
    min_vol_pct    = np.float32(0.001)

    for i in range(n - max_hold):
        # We compute features using data up to bar 'i'. 
        # But our simulated trade must enter at the open of bar 'i+1'.
        entry_price = open_arr[i + 1]
        vol_dist    = stop_distances[i]

        if vol_dist <= 0 or entry_price <= 0:
            continue

        vol_pct = max(vol_dist / entry_price, min_vol_pct)

        # ── LONG ──────────────────────────────────────────────────────────────
        stop_level        = entry_price - vol_dist
        peak_price        = entry_price
        max_risk_consumed = np.float32(0.0)
        long_pnl_pct      = np.float32(0.0)

        for k in range(1, max_hold):
            idx = i + k
            c_open, c_high, c_low, c_close = (
                open_arr[idx], high_arr[idx], low_arr[idx], close_arr[idx]
            )

            if c_low <= stop_level:
                exit_price   = min(c_open, stop_level)
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed = np.float32(1.0)
                break

            drawdown_from_entry   = entry_price - c_low
            current_risk_consumed = drawdown_from_entry / vol_dist
            if current_risk_consumed > max_risk_consumed:
                max_risk_consumed = current_risk_consumed

            if c_high > peak_price:
                peak_price = c_high
                new_stop   = peak_price - vol_dist
                if new_stop > stop_level:
                    stop_level = new_stop

            if k == max_hold - 1:
                long_pnl_pct = (c_close - entry_price) / entry_price

        # ── SHORT ─────────────────────────────────────────────────────────────
        stop_level_short        = entry_price + vol_dist
        trough_price            = entry_price
        max_risk_consumed_short = np.float32(0.0)
        short_pnl_pct           = np.float32(0.0)

        for k in range(1, max_hold):
            idx = i + k
            c_open, c_high, c_low, c_close = (
                open_arr[idx], high_arr[idx], low_arr[idx], close_arr[idx]
            )

            if c_high >= stop_level_short:
                exit_price    = max(c_open, stop_level_short)
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = np.float32(1.0)
                break

            drawdown_from_entry   = c_high - entry_price
            current_risk_consumed = drawdown_from_entry / vol_dist
            if current_risk_consumed > max_risk_consumed_short:
                max_risk_consumed_short = current_risk_consumed

            if c_low < trough_price:
                trough_price = c_low
                new_stop     = trough_price + vol_dist
                if new_stop < stop_level_short:
                    stop_level_short = new_stop

            if k == max_hold - 1:
                short_pnl_pct = (entry_price - c_close) / entry_price

        # ── SCORING ───────────────────────────────────────────────────────────
        long_r  = long_pnl_pct  / vol_pct
        short_r = short_pnl_pct / vol_pct

        max_risk_consumed       = min(max(max_risk_consumed, np.float32(0.0)), np.float32(1.0))
        max_risk_consumed_short = min(max(max_risk_consumed_short, np.float32(0.0)), np.float32(1.0))

        cost_r      = total_cost_pct / vol_pct
        long_r_net  = long_r  - cost_r
        short_r_net = short_r - cost_r

        if long_r_net > 0 and long_r_net > short_r_net:
            targets[i] =  np.tanh(long_r_net  / saturation_factor)
        elif short_r_net > 0 and short_r_net > long_r_net:
            targets[i] = -np.tanh(short_r_net / saturation_factor)

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_oracle_targets(
    df_raw, 
    df_features,
    max_hold:   int   = 96,
    atr_period: int   = 14,
    atr_mult:   float = 1.4,
    fee_per_side: float = 0.0003,
    slippage:     float = 0.0001,
):
    print(f"Generating targets using Oracle 4.0  "
          f"[Wilder ATR({atr_period}), mult={atr_mult}, max_hold={max_hold}, "
          f"fee_per_side={fee_per_side}, slippage={slippage}]...")

    # Step 1: Compute Wilder ATR in float64
    atr = _compute_wilder_atr(df_raw, period=atr_period)

    # Step 2: Drop NaN warmup rows
    valid_mask   = atr.notna()
    df_raw_valid = df_raw.loc[valid_mask]
    atr_valid    = atr.loc[valid_mask]

    warmup_dropped = (~valid_mask).sum()
    print(f"  ATR warmup rows dropped: {warmup_dropped} (first {atr_period - 1} bars)")

    # Step 3: Run Numba scoring strictly downcast to float32
    # Numba will crash if we accidentally pass float64 here, enforcing memory limits.
    targets = run_oracle_scoring(
        df_raw_valid['open'].values.astype(np.float32),
        df_raw_valid['high'].values.astype(np.float32),
        df_raw_valid['low'].values.astype(np.float32),
        df_raw_valid['close'].values.astype(np.float32),
        atr_valid.values.astype(np.float32),
        np.int64(max_hold),
        np.float32(fee_per_side),
        np.float32(slippage),
        np.float32(atr_mult),
        np.float32(2.0),    # saturation_factor
        np.float32(0.25)    # mae_penalty
    )

    y_targets = pd.Series(
        targets, index=df_raw_valid.index, name='target', dtype=np.float32
    )

    # Step 4: Align on common index and trim zero-padded tail
    common_index = df_features.index.intersection(y_targets.index)
    common_index = common_index[:-max_hold]

    df_features = df_features.loc[common_index]
    y_targets   = y_targets.loc[common_index]

    # Step 5: Label distribution diagnostic
    n_long  = (y_targets > 0).sum()
    n_short = (y_targets < 0).sum()
    n_flat  = (y_targets == 0).sum()
    print(
        f"Target generation complete. Scored rows: {len(common_index)}\n"
        f"  Label distribution → Long: {n_long} | Short: {n_short} | Flat: {n_flat}"
    )

    return df_features, y_targets
