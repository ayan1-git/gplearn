import importlib
import numpy as np
import numba
import pandas as pd

try:
    config = importlib.import_module("src.config")
    DEFAULT_MAX_HOLD = int(config.ORACLE_MAX_HOLD)
    DEFAULT_BARRIER_MULT = float(config.ORACLE_ATR_MULT)
except (ImportError, AttributeError):
    DEFAULT_MAX_HOLD = 96
    DEFAULT_BARRIER_MULT = 2.0


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
# NUMBA CORE - CONTINUOUS TRIPLE BARRIER
# ─────────────────────────────────────────────────────────────────────────────

@numba.jit(
    "f4[:](f4[:], f4[:], f4[:], f4[:], f4[:], i8, f4)",
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
    barrier_mult
):
    """
    Continuous Triple-Barrier Method.

    Decision time: bar i
    Execution anchor: next bar open, open_arr[i + 1]

    Labels:
    +1.0  -> upper barrier hit first
    -1.0  -> lower barrier hit first
    (-0.99, 0.99) -> vertical-barrier timeout, normalized by barrier distance
    0.0   -> ambiguous double-touch bar (both barriers breached in same OHLC bar)

    Returns target aligned to decision bar i.
    """
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    if max_hold <= 0:
        return targets

    # Need room for:
    # - next-bar entry at i + 1
    # - full holding window up to i + max_hold
    for i in range(n - max_hold):
        entry_idx = i + 1
        entry_price = open_arr[entry_idx]
        vol_dist = atr_arr[i] * barrier_mult

        if not np.isfinite(entry_price) or not np.isfinite(vol_dist):
            continue
        if entry_price <= 0.0 or vol_dist <= 0.0:
            continue

        upper_barrier = entry_price + vol_dist
        lower_barrier = entry_price - vol_dist

        resolved = False

        # Inspect each future bar, including the entry bar itself.
        for idx in range(entry_idx, i + max_hold + 1):
            c_high = high_arr[idx]
            c_low = low_arr[idx]

            if not np.isfinite(c_high) or not np.isfinite(c_low):
                continue

            hit_upper = c_high >= upper_barrier
            hit_lower = c_low <= lower_barrier

            if hit_upper and hit_lower:
                # OHLC does not reveal intrabar ordering. Stay neutral.
                targets[i] = np.float32(0.0)
                resolved = True
                break
            elif hit_upper:
                targets[i] = np.float32(1.0)
                resolved = True
                break
            elif hit_lower:
                targets[i] = np.float32(-1.0)
                resolved = True
                break

        if not resolved:
            final_close = close_arr[i + max_hold]
            if np.isfinite(final_close):
                frac_return = (final_close - entry_price) / vol_dist
                if frac_return > 0.99:
                    frac_return = 0.99
                elif frac_return < -0.99:
                    frac_return = -0.99
                targets[i] = np.float32(frac_return)

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_tbm_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int = DEFAULT_MAX_HOLD,
    atr_period: int = 14,
    barrier_mult: float = DEFAULT_BARRIER_MULT,
):
    """
    Generate continuous next-open TBM targets aligned to df_features.

    Parameters
    ----------
    df_raw : DataFrame
        Must contain open, high, low, close columns.
    df_features : DataFrame
        Feature matrix already computed from the same raw index.
    max_hold : int
        Maximum holding horizon in bars.
    atr_period : int
        Wilder ATR period.
    barrier_mult : float
        Symmetric barrier size in ATR units.

    Returns
    -------
    df_features_aligned : DataFrame
    y_targets : Series[float32]
    """
    if max_hold <= 0:
        raise ValueError("max_hold must be > 0")
    if barrier_mult <= 0:
        raise ValueError("barrier_mult must be > 0")
    if not {"open", "high", "low", "close"}.issubset(set(df_raw.columns)):
        raise ValueError("df_raw must contain open, high, low, close columns")

    print(
        f"Generating Continuous TBM targets "
        f"[ATR({atr_period}), barrier_mult={barrier_mult}, max_hold={max_hold}]..."
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
        np.float32(barrier_mult),
    )

    y_targets = pd.Series(
        targets,
        index=df_raw_valid.index,
        name="target",
        dtype=np.float32
    )

    common_index = df_features.index.intersection(y_targets.index)
    if len(common_index) <= max_hold:
        raise ValueError(
            "Not enough aligned rows after ATR warmup and feature intersection "
            "to trim the non-computable TBM tail."
        )

    decision_index = common_index[:-max_hold]

    df_features_aligned = df_features.loc[decision_index]
    y_targets_aligned = y_targets.loc[decision_index]

    n_long = int((y_targets_aligned > 0).sum())
    n_short = int((y_targets_aligned < 0).sum())
    n_flat = int((y_targets_aligned == 0).sum())

    print(
        f"Target generation complete. Scored rows: {len(decision_index)}\n"
        f"  Label distribution → Long: {n_long} | Short: {n_short} | Flat: {n_flat}"
    )

    return df_features_aligned, y_targets_aligned


# Backward-compatible alias so older imports do not break immediately.
def generate_oracle_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int = DEFAULT_MAX_HOLD,
    atr_period: int = 14,
    atr_mult: float = DEFAULT_BARRIER_MULT,
):
    return generate_tbm_targets(
        df_raw=df_raw,
        df_features=df_features,
        max_hold=max_hold,
        atr_period=atr_period,
        barrier_mult=atr_mult,
    )
