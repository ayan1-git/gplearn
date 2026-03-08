import gc
import importlib
import os
from typing import Tuple

import numpy as np
import pandas as pd

# --- MODULE IMPORTS ---
cfg = importlib.import_module("src.config")
fe = importlib.import_module("src.01_feature_engineering")
tg = importlib.import_module("src.02_target_generator")
gp = importlib.import_module("src.03_gp_engine")
vbteval = importlib.import_module("src.04_vectorbt_evaluator")

calculate_features = fe.calculate_features
generate_tbm_targets = tg.generate_tbm_targets
train_gp_model = gp.train_gp_model
evaluate_formula_with_vectorbt = vbteval.evaluate_formula_with_vectorbt

PASSTHROUGH_FEATURES = fe.PASSTHROUGH_FEATURES
SCALE_FEATURES = fe.SCALE_FEATURES

# Config Parameters
ORACLE_MAX_HOLD = cfg.ORACLE_MAX_HOLD
TP_ATR_MULT = cfg.TP_ATR_MULT
SL_ATR_MULT = cfg.SL_ATR_MULT
ENTRY_PCT = cfg.ENTRY_PCT
EXIT_PCT = cfg.EXIT_PCT
TRAIN_MONTHS = cfg.TRAIN_MONTHS
TEST_MONTHS = cfg.TEST_MONTHS
DATAPATH = cfg.DATAPATH


def setup_directories() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/vectorbt_stats", exist_ok=True)


def _validate_ohlc(df_raw: pd.DataFrame) -> None:
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Raw data missing required OHLC columns: {sorted(missing)}")

    if not df_raw.index.is_monotonic_increasing:
        raise ValueError("Raw data index must be sorted ascending.")

    if df_raw.index.has_duplicates:
        raise ValueError("Raw data index contains duplicate timestamps.")


def _validate_feature_alignment(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    y_targets: pd.Series,
) -> None:
    if len(df_features) == 0:
        raise ValueError("Feature frame is empty after preprocessing.")
    if len(y_targets) == 0:
        raise ValueError("Target series is empty after target generation.")
    if not df_features.index.equals(y_targets.index):
        raise ValueError("Feature and target indices are not perfectly aligned.")
    if not df_raw.index.equals(df_features.index):
        raise ValueError("Raw OHLC data is not aligned to feature index.")
    if y_targets.isna().any():
        raise ValueError("Target series contains NaNs after alignment.")
    if not np.isfinite(y_targets.to_numpy(dtype=np.float32)).all():
        raise ValueError("Target series contains non-finite values.")


def load_and_prepare_data(filepath: str):
    print(f"Loading raw data from {filepath}...")
    df_raw = pd.read_csv(filepath, parse_dates=["datetime"], index_col="datetime")
    df_raw.sort_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]

    _validate_ohlc(df_raw)

    feature_kwargs = dict(
        add_session_features=True,
        session=fe.SessionConfig(open_time="09:15", close_time="15:30", tz="Asia/Kolkata"),
        clip_outside_session=True,
        mds_fast_window=5,
        mds_slow_window=30,
        vol_asym_window=20,
    )

    df_features = calculate_features(df_raw, **feature_kwargs)
    print(f"Features ready. Columns: {list(df_features.columns)}")

    df_features, y_targets = generate_tbm_targets(
        df_raw=df_raw,
        df_features=df_features,
        max_hold=ORACLE_MAX_HOLD,
        tp_mult=TP_ATR_MULT,
        sl_mult=SL_ATR_MULT,
        target_mode="first_touch_class",
        timeout_policy="neutral",
        ambiguity_policy="neutral",
        drop_invalid=True,
        return_metadata=False,
    )

    df_features = df_features.astype(np.float32)
    y_targets = y_targets.astype(np.float32)
    df_raw = df_raw.loc[df_features.index].astype(np.float32)

    _validate_feature_alignment(df_raw, df_features, y_targets)

    n_long = int((y_targets == 1.0).sum())
    n_short = int((y_targets == -1.0).sum())
    n_neutral = int((y_targets == 0.0).sum())

    print(
        "Prepared aligned dataset.\n"
        f"  Rows: {len(df_features)} | Features: {df_features.shape[1]}\n"
        f"  Targets -> Long: {n_long} | Short: {n_short} | Neutral: {n_neutral}"
    )

    return df_raw, df_features, y_targets


def tanh_scale_train_apply_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_cols: list,
    passthrough_cols: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zero-preserving tanh scaling fit on train only.
    """
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train and test feature frames must both be non-empty.")

    original_columns = list(X_train.columns)
    actual_scale_cols = [c for c in scale_cols if c in X_train.columns]
    actual_pass_cols = [c for c in passthrough_cols if c in X_train.columns]

    missing_output_cols = [c for c in original_columns if c not in set(actual_scale_cols + actual_pass_cols)]
    if missing_output_cols:
        raise ValueError(
            f"Some columns are neither in SCALE_FEATURES nor PASSTHROUGH_FEATURES: {missing_output_cols}"
        )

    X_train_scaled = pd.DataFrame(index=X_train.index)
    X_test_scaled = pd.DataFrame(index=X_test.index)
    eps = 1e-8

    if actual_scale_cols:
        train_scale_view = X_train[actual_scale_cols].astype(np.float32)
        test_scale_view = X_test[actual_scale_cols].astype(np.float32)

        scale_factors = np.percentile(np.abs(train_scale_view), 75, axis=0)
        scale_factors = np.maximum(scale_factors, eps)

        X_train_scaled[actual_scale_cols] = np.tanh(train_scale_view / scale_factors).astype(np.float32)
        X_test_scaled[actual_scale_cols] = np.tanh(test_scale_view / scale_factors).astype(np.float32)

    if actual_pass_cols:
        X_train_scaled[actual_pass_cols] = X_train[actual_pass_cols].astype(np.float32)
        X_test_scaled[actual_pass_cols] = X_test[actual_pass_cols].astype(np.float32)

    X_train_scaled = X_train_scaled[original_columns]
    X_test_scaled = X_test_scaled[original_columns]

    if X_train_scaled.isna().any().any() or X_test_scaled.isna().any().any():
        raise ValueError("NaNs detected after train/test scaling.")

    return X_train_scaled, X_test_scaled


def _extract_fold_data(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    y_targets: pd.Series,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
):
    train_end_inclusive = train_end - pd.Timedelta(nanoseconds=1)

    X_train_raw = df_features.loc[train_start:train_end_inclusive]
    y_train_raw = y_targets.loc[train_start:train_end_inclusive]

    X_test_raw = df_features.loc[train_end:test_end]
    raw_test = df_raw.loc[train_end:test_end]

    if not X_train_raw.index.equals(y_train_raw.index):
        raise ValueError("Training feature/target indices are misaligned inside fold extraction.")

    if not X_test_raw.index.equals(raw_test.index):
        raise ValueError("Test feature/raw indices are misaligned inside fold extraction.")

    return X_train_raw, y_train_raw, X_test_raw, raw_test


def walk_forward_optimization(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    y_targets: pd.Series,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
    data_path: str = "Unknown",
):
    print("=" * 50)
    print("STARTING WALK-FORWARD OPTIMIZATION")
    print("=" * 50)

    start_date = df_features.index.min()
    end_date = df_features.index.max()

    current_train_start = start_date
    fold = 1
    winning_formulas = []

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            print("Reached end of dataset. WFO complete.")
            break

        print(f"\n--- FOLD {fold} ---")
        print(f"Train Window: {current_train_start.date()} to {train_end.date()}")
        print(f"Test Window: {train_end.date()} to {test_end.date()}")

        try:
            X_train_raw, y_train_raw, X_test_raw, raw_test = _extract_fold_data(
                df_raw=df_raw,
                df_features=df_features,
                y_targets=y_targets,
                train_start=current_train_start,
                train_end=train_end,
                test_end=test_end,
            )
        except Exception as e:
            print(f"-> Fold {fold} extraction failed: {e}")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        # IMPORTANT:
        # Do NOT trim ORACLE_MAX_HOLD again here.
        # generate_tbm_targets() has already removed the final max_hold decision rows globally.
        if len(X_train_raw) < 500 or len(X_test_raw) < 200:
            print(
                f"Not enough data in this fold. "
                f"Train rows: {len(X_train_raw)} | Test rows: {len(X_test_raw)}. Skipping."
            )
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        if y_train_raw.nunique(dropna=True) < 2:
            print("Training targets have fewer than 2 unique classes. Skipping fold.")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        try:
            X_train, X_test = tanh_scale_train_apply_test(
                X_train=X_train_raw,
                X_test=X_test_raw,
                scale_cols=SCALE_FEATURES,
                passthrough_cols=PASSTHROUGH_FEATURES,
            )
            y_train = y_train_raw.astype(np.float32)

            if not X_train.index.equals(y_train.index):
                raise ValueError("Scaled training features and targets are misaligned.")
            if not X_test.index.equals(raw_test.index):
                raise ValueError("Scaled OOS features and raw OOS prices are misaligned.")
        except Exception as e:
            print(f"-> Fold {fold} preprocessing failed: {e}")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        try:
            gp_model = train_gp_model(X_train, y_train)
            formula_str = str(gp_model._program)

            train_outputs = gp_model.predict(X_train.values)
            train_min = float(np.nanmin(train_outputs))
            train_max = float(np.nanmax(train_outputs))

            print(
                f"-> Formula Seed: {gp_model.random_state} | "
                f"Train Score Range: [{train_min:.3f}, {train_max:.3f}]"
            )
            print(f"-> Logic: {formula_str}")

            train_buy = np.percentile(train_outputs, ENTRY_PCT)
            train_sell = np.percentile(train_outputs, EXIT_PCT)

            train_long_mask = (train_outputs > train_buy) & (train_outputs >= cfg.ABSOLUTE_EDGE_FLOOR)
            train_short_mask = (train_outputs < train_sell) & (train_outputs <= -cfg.ABSOLUTE_EDGE_FLOOR)

            n_train_long = int(train_long_mask.sum())
            n_train_short = int(train_short_mask.sum())
            n_train_trades = n_train_long + n_train_short

            long_share = n_train_long / max(n_train_trades, 1)
            short_share = n_train_short / max(n_train_trades, 1)

            print(
                f"-> Train signal mix | Longs: {n_train_long} | Shorts: {n_train_short} | "
                f"Long share: {long_share:.2%} | Short share: {short_share:.2%}"
            )
        except Exception as e:
            print(f"GP training failed on fold {fold}: {e}")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        try:
            portfolio, stats, metadata = evaluate_formula_with_vectorbt(
                gp_model=gp_model,
                df_features_oos=X_test,
                df_raw_oos=raw_test,
                long_pct_level=ENTRY_PCT,
                short_pct_level=EXIT_PCT,
                tp_mult=TP_ATR_MULT,
                sl_mult=SL_ATR_MULT,
            )
        except Exception as e:
            print(f"VectorBT evaluation failed on fold {fold}: {e}")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            del X_train_raw, y_train_raw, X_test_raw, raw_test, X_train, X_test, y_train, gp_model
            gc.collect()
            continue

        total_return = stats.get("Total Return [%]", 0.0)
        sharpe = stats.get("Sharpe Ratio", 0.0)

        if pd.isna(total_return):
            total_return = 0.0
        if pd.isna(sharpe):
            sharpe = 0.0

        if total_return > 0 and sharpe > 0.5:
            print(f"-> SUCCESS! Formula survived OOS. Ret: {total_return:.2f}%, Sharpe: {sharpe:.2f}")

            winning_formulas.append(
                {
                    "fold": fold,
                    "formula": formula_str,
                    "return_pct": float(total_return),
                    "sharpe": float(sharpe),
                    "tp_mult": float(TP_ATR_MULT),
                    "sl_mult": float(SL_ATR_MULT),
                    "coverage_pct": float(metadata["coverage_pct"]),
                    "n_long": int(metadata["n_long"]),
                    "n_short": int(metadata["n_short"]),
                }
            )

            stats.to_frame(name="value").to_csv(f"outputs/vectorbt_stats/fold_{fold}_winner.csv")
        else:
            print(f"-> FAILED. Formula collapsed in OOS. Ret: {total_return:.2f}%, Sharpe: {sharpe:.2f}")

        current_train_start += pd.DateOffset(months=test_months)
        fold += 1

        del X_train_raw, y_train_raw, X_test_raw, raw_test, X_train, X_test, y_train
        del gp_model, portfolio, stats, metadata
        gc.collect()

    print("=" * 50)
    print("DISCOVERY RUN COMPLETE")
    print("=" * 50)

    if winning_formulas:
        print(f"Found {len(winning_formulas)} robust strategies.")
        log_file = "outputs/winning_formulas.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Discovery Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {data_path}\n")
            f.write(
                f"Parameters: TP_MULT={TP_ATR_MULT}, SL_MULT={SL_ATR_MULT}, "
                f"MAX_HOLD={ORACLE_MAX_HOLD}\n"
            )
            f.write("-" * 60 + "\n")
            for w in winning_formulas:
                f.write(
                    f"Fold {w['fold']} | Ret: {w['return_pct']:.2f}% | "
                    f"Sharpe: {w['sharpe']:.2f} | Coverage: {w['coverage_pct']:.2f}%\n"
                )
                f.write(
                    f"TP/SL Ratio: {w['tp_mult']}/{w['sl_mult']} | "
                    f"Signals: L={w['n_long']}, S={w['n_short']}\n"
                )
                f.write(f"Logic: {w['formula']}\n\n")
    else:
        print("No robust strategies found.")


if __name__ == "__main__":
    setup_directories()

    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(f"Data file not found at {DATAPATH}")

    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        walk_forward_optimization(
            df_raw=df_raw,
            df_features=df_features,
            y_targets=y_targets,
            data_path=DATAPATH,
        )
    except Exception as e:
        print(f"Pipeline crashed: {e}")
        raise
