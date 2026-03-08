import importlib
import numpy as np
import numba
import pandas as pd

try:
    config = importlib.import_module("src.config")
    DEFAULT_MAX_HOLD = int(config.ORACLE_MAX_HOLD)
    DEFAULT_TP_MULT = float(config.TP_ATR_MULT)
    DEFAULT_SL_MULT = float(config.SL_ATR_MULT)
except (ImportError, AttributeError):
    DEFAULT_MAX_HOLD = 96
    DEFAULT_TP_MULT = 2.0
    DEFAULT_SL_MULT = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE UTILITY: Wilder's ATR (RMA)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wilder_atr(df_raw: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Wilder ATR using float64 arithmetic for numerical stability.
    """
    high_low = df_raw["high"] - df_raw["low"]
    high_close = (df_raw["high"] - df_raw["close"].shift(1)).abs()
    low_close = (df_raw["low"] - df_raw["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return atr


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA CORE - ASYMMETRIC CONTINUOUS TRIPLE BARRIER
# ─────────────────────────────────────────────────────────────────────────────

@numba.jit(
    "f4[:](f4[:], f4[:], f4[:], f4[:], f4[:], i8, f4, f4)",
    nopython=True,
    cache=True,
    fastmath=True
)
def run_continuous_triple_barrier(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    max_hold,
    tp_mult,
    sl_mult
):
    """
    Asymmetric Continuous Triple-Barrier Method.

    Decision time: bar i
    Execution anchor: next bar open, open_arr[i + 1]

    Labels:
    +1.0  -> Long TP hit before Long SL
    -1.0  -> Short TP hit before Short SL
    0.0   -> Ambiguous (both hits or neither in a way that creates no bias)
    """
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    if max_hold <= 0:
        return targets

    for i in range(n - max_hold):
        entry_idx = i + 1
        entry_price = open_arr[entry_idx]
        atr_val = atr_arr[i]
        
        if not np.isfinite(entry_price) or not np.isfinite(atr_val):
            continue
        if entry_price <= 0.0 or atr_val <= 0.0:
            continue

        tp_dist = atr_val * tp_mult
        sl_dist = atr_val * sl_mult

        # Long Thresholds
        l_tp = entry_price + tp_dist
        l_sl = entry_price - sl_dist

        # Short Thresholds
        s_tp = entry_price - tp_dist
        s_sl = entry_price + sl_dist

        # Resolution status
        # 1  = Success (TP hit first)
        # -1 = Failure (SL hit first)
        res_l = 0
        res_s = 0

        for idx in range(entry_idx, i + max_hold + 1):
            c_high = high_arr[idx]
            c_low = low_arr[idx]

            if not np.isfinite(c_high) or not np.isfinite(c_low):
                continue

            # Check Long
            if res_l == 0:
                hit_tp = c_high >= l_tp
                hit_sl = c_low <= l_sl
                if hit_tp and hit_sl:
                    res_l = -2 # Ambiguous bar
                elif hit_tp:
                    res_l = 1
                elif hit_sl:
                    res_l = -1

            # Check Short
            if res_s == 0:
                hit_tp = c_low <= s_tp
                hit_sl = c_high >= s_sl
                if hit_tp and hit_sl:
                    res_s = -2 # Ambiguous bar
                elif hit_tp:
                    res_s = 1
                elif hit_sl:
                    res_s = -1

            if res_l != 0 and res_s != 0:
                break

        # Final Labeling Logic
        if res_l == 1 and res_s != 1:
            targets[i] = np.float32(1.0)
        elif res_s == 1 and res_l != 1:
            targets[i] = np.float32(-1.0)
        elif res_l == 1 and res_s == 1:
            # Both succeeded? Pick the one with larger TP if different, 
            # but usually we just stay neutral.
            targets[i] = np.float32(0.0)
        else:
            # Timeout or Failure cases
            # Use path-dependent return normalized by TP distance
            final_close = close_arr[i + max_hold]
            if np.isfinite(final_close):
                # Calculate return relative to the 'success' threshold
                raw_ret = (final_close - entry_price) / tp_dist
                # Clip to [-0.99, 0.99] to differentiate from hard barrier hits
                if raw_ret > 0.99: raw_ret = 0.99
                elif raw_ret < -0.99: raw_ret = -0.99
                targets[i] = np.float32(raw_ret)

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_tbm_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int = DEFAULT_MAX_HOLD,
    atr_period: int = 14,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
):
    """
    Generate asymmetric next-open TBM targets aligned to df_features.
    """
    if max_hold <= 0:
        raise ValueError("max_hold must be > 0")
    if tp_mult <= 0 or sl_mult <= 0:
        raise ValueError("Multipliers must be > 0")
    if not {"open", "high", "low", "close"}.issubset(set(df_raw.columns)):
        raise ValueError("df_raw must contain open, high, low, close columns")

    print(
        f"Generating Asymmetric TBM targets\n"
        f"  [ATR({atr_period}), TP_mult={tp_mult}, SL_mult={sl_mult}, max_hold={max_hold}]"
    )

    atr = _compute_wilder_atr(df_raw, period=atr_period)

    valid_mask = atr.notna()
    df_raw_valid = df_raw.loc[valid_mask]
    atr_valid = atr.loc[valid_mask]

    warmup_dropped = int((~valid_mask).sum())
    print(f"  ATR warmup rows dropped: {warmup_dropped} (first {atr_period - 1} bars)")

    targets = run_continuous_triple_barrier(
        df_raw_valid["open"].to_numpy(dtype=np.float32),
        df_raw_valid["high"].to_numpy(dtype=np.float32),
        df_raw_valid["low"].to_numpy(dtype=np.float32),
        df_raw_valid["close"].to_numpy(dtype=np.float32),
        atr_valid.to_numpy(dtype=np.float32),
        np.int64(max_hold),
        np.float32(tp_mult),
        np.float32(sl_mult),
    )

    y_targets = pd.Series(
        targets,
        index=df_raw_valid.index,
        name="target",
        dtype=np.float32
    )

    common_index = df_features.index.intersection(y_targets.index)
    if len(common_index) <= max_hold:
        raise ValueError("Not enough aligned rows.")

    decision_index = common_index[:-max_hold]
    df_features_aligned = df_features.loc[decision_index]
    y_targets_aligned = y_targets.loc[decision_index]

    n_long = int((y_targets_aligned > 0.5).sum())
    n_short = int((y_targets_aligned < -0.5).sum())
    
    print(
        f"Target generation complete. Scored rows: {len(decision_index)}\n"
        f"  Label distribution → Long Hits: {n_long} | Short Hits: {n_short}"
    )

    return df_features_aligned, y_targets_aligned


# Backward-compatible alias
def generate_oracle_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int = DEFAULT_MAX_HOLD,
    atr_period: int = 14,
    atr_mult: float = DEFAULT_TP_MULT, # Defaulting to TP for backward compatibility
):
    return generate_tbm_targets(
        df_raw=df_raw,
        df_features=df_features,
        max_hold=max_hold,
        atr_period=atr_period,
        tp_mult=atr_mult,
        sl_mult=atr_mult, # Symmetric if only one mult provided
    )
