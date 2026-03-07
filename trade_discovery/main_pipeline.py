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

ORACLE_MAX_HOLD = cfg.ORACLE_MAX_HOLD
ORACLE_ATR_MULT = cfg.ORACLE_ATR_MULT
ENTRY_PCT = cfg.ENTRY_PCT
EXIT_PCT = cfg.EXIT_PCT
TRAIN_MONTHS = cfg.TRAIN_MONTHS
TEST_MONTHS = cfg.TEST_MONTHS
DATAPATH = cfg.DATAPATH


def setup_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/vectorbt_stats", exist_ok=True)


def load_and_prepare_data(filepath: str):
    print(f"Loading raw data from {filepath}...")
    df_raw = pd.read_csv(filepath, parse_dates=["datetime"], index_col="datetime")
    df_raw.sort_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]

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
        barrier_mult=ORACLE_ATR_MULT,
    )

    df_raw = df_raw.loc[df_features.index].astype(np.float32)

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
    original_columns = X_train.columns
    actual_scale_cols = [c for c in scale_cols if c in X_train.columns]
    actual_pass_cols = [c for c in passthrough_cols if c in X_train.columns]

    X_train_scaled = pd.DataFrame(index=X_train.index)
    X_test_scaled = pd.DataFrame(index=X_test.index)

    eps = 1e-8

    if actual_scale_cols:
        train_scale_view = X_train[actual_scale_cols]
        test_scale_view = X_test[actual_scale_cols]

        scale_factors = np.percentile(np.abs(train_scale_view), 75, axis=0)
        scale_factors = np.maximum(scale_factors, eps)

        X_train_scaled[actual_scale_cols] = np.tanh(
            train_scale_view / scale_factors
        ).astype(np.float32)

        X_test_scaled[actual_scale_cols] = np.tanh(
            test_scale_view / scale_factors
        ).astype(np.float32)

    if actual_pass_cols:
        X_train_scaled[actual_pass_cols] = X_train[actual_pass_cols].astype(np.float32)
        X_test_scaled[actual_pass_cols] = X_test[actual_pass_cols].astype(np.float32)

    return X_train_scaled[original_columns], X_test_scaled[original_columns]


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

        train_end_inclusive = train_end - pd.Timedelta(nanoseconds=1)

        X_train_raw = df_features.loc[current_train_start:train_end_inclusive]
        y_train_raw = y_targets.loc[current_train_start:train_end_inclusive]

        X_test_raw = df_features.loc[train_end:test_end]
        raw_test = df_raw.loc[train_end:test_end]

        if len(X_train_raw) > ORACLE_MAX_HOLD:
            X_train_raw = X_train_raw.iloc[:-ORACLE_MAX_HOLD]
            y_train_raw = y_train_raw.iloc[:-ORACLE_MAX_HOLD]
        else:
            print(f"-> Fold {fold} training set too short to purge. Skipping.")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        if len(X_train_raw) < 500 or len(X_test_raw) < 200:
            print("Not enough data in this fold after purging. Skipping.")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        X_train, X_test = tanh_scale_train_apply_test(
            X_train=X_train_raw,
            X_test=X_test_raw,
            scale_cols=SCALE_FEATURES,
            passthrough_cols=PASSTHROUGH_FEATURES,
        )
        y_train = y_train_raw.astype(np.float32)

        try:
            gp_model = train_gp_model(X_train, y_train)
            formula_str = str(gp_model._program)

            train_scores = pd.Series(
                gp_model.predict(X_train.values),
                index=X_train.index,
                name="train_score"
            )

            print(
                "-> Train score diagnostics | "
                f"mean: {train_scores.mean():.6f} | "
                f"std: {train_scores.std(ddof=0):.6f} | "
                f"min: {train_scores.min():.6f} | "
                f"max: {train_scores.max():.6f}"
            )
            print(
                f"-> Using rank thresholds from config | "
                f"Long >= {ENTRY_PCT}th percentile, Short <= {EXIT_PCT}th percentile"
            )

        except Exception as e:
            print(f"GP training failed on fold {fold}: {e}")
            current_train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        portfolio, stats, metadata = evaluate_formula_with_vectorbt(
            gp_model=gp_model,
            df_features_oos=X_test,
            df_raw_oos=raw_test,
            long_pct_level=ENTRY_PCT,
            short_pct_level=EXIT_PCT,
        )

        total_return = stats.get("Total Return [%]", 0.0)
        sharpe = stats.get("Sharpe Ratio", 0.0)

        if pd.isna(total_return):
            total_return = 0.0
        if pd.isna(sharpe):
            sharpe = 0.0

        if total_return > 0 and sharpe > 0.5:
            print(
                f"-> SUCCESS! Formula survived OOS. "
                f"Return: {total_return:.2f}%, Sharpe: {sharpe:.2f}"
            )

            winning_formulas.append(
                {
                    "fold": fold,
                    "formula": formula_str,
                    "return_pct": float(total_return),
                    "sharpe": float(sharpe),
                    "long_pct_level": float(ENTRY_PCT),
                    "short_pct_level": float(EXIT_PCT),
                    "coverage_pct": float(metadata["coverage_pct"]),
                    "n_long": int(metadata["n_long"]),
                    "n_short": int(metadata["n_short"]),
                }
            )

            if hasattr(stats, "to_frame"):
                stats.to_frame(name="value").to_csv(
                    f"outputs/vectorbt_stats/fold_{fold}_winner.csv"
                )
            else:
                pd.DataFrame(stats).to_csv(
                    f"outputs/vectorbt_stats/fold_{fold}_winner.csv"
                )
        else:
            print(
                f"-> FAILED. Formula collapsed in OOS. "
                f"Return: {total_return:.2f}%, Sharpe: {sharpe:.2f}"
            )
            print("Discarding formula and moving to next fold.")

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
                f"Parameters: TBM_ATR_MULT={ORACLE_ATR_MULT}, "
                f"MAX_HOLD={ORACLE_MAX_HOLD}, ENTRY_PCT={ENTRY_PCT}, EXIT_PCT={EXIT_PCT}\n"
            )
            f.write("-" * 60 + "\n")
            for w in winning_formulas:
                f.write(
                    f"Fold {w['fold']} | Ret: {w['return_pct']:.2f}% | "
                    f"Sharpe: {w['sharpe']:.2f} | Coverage: {w['coverage_pct']:.2f}%\n"
                )
                f.write(
                    f"Signals: Longs={w['n_long']} | Shorts={w['n_short']} | "
                    f"Thresholds: Long>={w['long_pct_level']}pct, "
                    f"Short<={w['short_pct_level']}pct\n"
                )
                f.write(f"Logic: {w['formula']}\n\n")
        print("Winners appended to logfile.")
    else:
        print("No robust strategies found. Consider adjusting parameters or providing more data.")


if __name__ == "__main__":
    setup_directories()

    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Data file not found at {DATAPATH}. "
            "Please check your filename and directory."
        )

    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        walk_forward_optimization(
            df_raw=df_raw,
            df_features=df_features,
            y_targets=y_targets,
            train_months=TRAIN_MONTHS,
            test_months=TEST_MONTHS,
            data_path=DATAPATH,
        )
    except Exception as e:
        print(f"Pipeline crashed: {e}")
        raise
