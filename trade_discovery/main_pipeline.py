import gc
import importlib
import logging
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------
ORACLE_MAX_HOLD = 52
ORACLE_ATR_MULT = 1.4

TRAIN_RATIO    = 0.70   # 70% train
TEST_RATIO     = 0.20   # 20% OOS test  (seen after each generation)
HOLDOUT_RATIO  = 0.10   # 10% holdout   (touched ONCE at the very end)

MAX_GENERATIONS        = 15    # maximum evolutionary generations
EARLY_STOP_PATIENCE    = 3     # stop if TEST sharpe drops for N consecutive gens
MIN_SHARPE_TO_SURVIVE  = 1.5   # per-generation survival floor
MIN_RETURN_TO_SURVIVE  = 2.0
MAX_DD_TO_SURVIVE      = 15.0

# ---------------------------------------------------------------------------
# MODULE IMPORTS
# ---------------------------------------------------------------------------
fe      = importlib.import_module("src.01_feature_engineering")
tg      = importlib.import_module("src.02_target_generator")
gp      = importlib.import_module("src.03_gp_engine")
vbteval = importlib.import_module("src.04_vectorbt_evaluator")

calculate_features             = fe.calculate_features
generate_oracle_targets        = tg.generate_oracle_targets
train_gp_model                 = gp.train_gp_model
extract_elite_programs         = gp.extract_elite_programs
hash_formula                   = gp.hash_formula
evaluate_formula_with_vectorbt = vbteval.evaluate_formula_with_vectorbt

PASSTHROUGH_FEATURES = fe.PASSTHROUGH_FEATURES
SCALE_FEATURES       = fe.SCALE_FEATURES

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def setup_directories():
    for d in ["data", "outputs/vectorbt_stats",
              "outputs/seed_checkpoints", "outputs/generations"]:
        os.makedirs(d, exist_ok=True)


def load_and_prepare_data(filepath: str):
    logger.info("Loading raw data from %s", filepath)
    df_raw = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    df_raw.sort_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]

    feature_kwargs = dict(
        add_session_features=True,
        session=fe.SessionConfig(open_time="09:15", close_time="15:30", tz="Asia/Kolkata"),
        clip_outside_session=True,
        mds_fast_window=5, mds_slow_window=30, vol_asym_window=20,
    )
    df_features = calculate_features(df_raw, **feature_kwargs)
    logger.info("Features computed: %s", list(df_features.columns))

    df_features, y_targets = generate_oracle_targets(
        df_raw, df_features,
        max_hold=ORACLE_MAX_HOLD, atr_mult=ORACLE_ATR_MULT
    )
    df_raw = df_raw.loc[df_features.index].astype(np.float32)
    return df_raw, df_features, y_targets


def tanh_scale_train_apply_test(
    X_train, X_test, scale_cols, passthrough_cols
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zero-Preserving Tanh Scaling.
    Scale factors fitted on TRAIN only — applied to TEST and HOLDOUT.
    Returns (scaled_train, scaled_test) AND the scale_factors dict
    so holdout can be scaled consistently.
    """
    original_columns  = X_train.columns
    actual_scale_cols = [c for c in scale_cols       if c in X_train.columns]
    actual_pass_cols  = [c for c in passthrough_cols if c in X_train.columns]
    EPS = 1e-8

    X_train_s = pd.DataFrame(index=X_train.index)
    X_test_s  = pd.DataFrame(index=X_test.index)

    scale_factors = {}
    if actual_scale_cols:
        sf = np.percentile(np.abs(X_train[actual_scale_cols]), 75, axis=0)
        sf = np.maximum(sf, EPS)
        scale_factors = dict(zip(actual_scale_cols, sf))
        X_train_s[actual_scale_cols] = np.tanh(X_train[actual_scale_cols] / sf).astype(np.float32)
        X_test_s[actual_scale_cols]  = np.tanh(X_test[actual_scale_cols]  / sf).astype(np.float32)
    if actual_pass_cols:
        X_train_s[actual_pass_cols] = X_train[actual_pass_cols].astype(np.float32)
        X_test_s[actual_pass_cols]  = X_test[actual_pass_cols].astype(np.float32)

    return X_train_s[original_columns], X_test_s[original_columns], scale_factors


def apply_scale_factors(X: pd.DataFrame, scale_factors: dict,
                        passthrough_cols: list) -> pd.DataFrame:
    """Apply pre-fitted scale factors to a new split (e.g., holdout)."""
    X_s = pd.DataFrame(index=X.index)
    for col in X.columns:
        if col in scale_factors:
            X_s[col] = np.tanh(X[col] / scale_factors[col]).astype(np.float32)
        else:
            X_s[col] = X[col].astype(np.float32)
    return X_s[X.columns]


# ---------------------------------------------------------------------------
# GENERATIONAL EVOLUTION ENGINE
# ---------------------------------------------------------------------------

def generational_evolution(
    df_raw, df_features, y_targets,
    data_path: str = "Unknown"
):
    """
    Generational Evolution: fixed train/test/holdout split.

    Each generation:
      1. Trains GP on full TRAIN (with seeds from prior generation)
      2. Evaluates best formula on TEST
      3. If survivor: extracts elite seeds for next generation
      4. Early stops if TEST sharpe degrades for EARLY_STOP_PATIENCE gens
      5. Resets seeds if formula fails TEST (bad genes not propagated)

    Final step: best formula from all generations validated on HOLDOUT.
    Holdout is touched EXACTLY ONCE.
    """
    logger.info("=" * 60)
    logger.info("GENERATIONAL EVOLUTION — Fixed Train/Test/Holdout Split")
    logger.info("Train: %.0f%% | Test: %.0f%% | Holdout: %.0f%%",
                TRAIN_RATIO * 100, TEST_RATIO * 100, HOLDOUT_RATIO * 100)
    logger.info("=" * 60)

    n = len(df_features)
    train_end_idx   = int(n * TRAIN_RATIO)
    test_end_idx    = int(n * (TRAIN_RATIO + TEST_RATIO))

    # Fixed chronological splits — NEVER shuffle time series
    idx_train   = df_features.index[:train_end_idx]
    idx_test    = df_features.index[train_end_idx:test_end_idx]
    idx_holdout = df_features.index[test_end_idx:]

    logger.info("Train  : %s → %s (%d bars)",
                idx_train[0].date(), idx_train[-1].date(), len(idx_train))
    logger.info("Test   : %s → %s (%d bars)",
                idx_test[0].date(), idx_test[-1].date(), len(idx_test))
    logger.info("Holdout: %s → %s (%d bars)",
                idx_holdout[0].date(), idx_holdout[-1].date(), len(idx_holdout))

    # --- Embargo purge on train end ---
    # Drop last ORACLE_MAX_HOLD bars from train to prevent label leakage
    X_train_raw = df_features.loc[idx_train].iloc[:-ORACLE_MAX_HOLD]
    y_train_raw = y_targets  .loc[idx_train].iloc[:-ORACLE_MAX_HOLD]
    X_test      = df_features.loc[idx_test]
    raw_test    = df_raw     .loc[idx_test]
    X_holdout   = df_features.loc[idx_holdout]
    raw_holdout = df_raw     .loc[idx_holdout]

    # Scale ONCE on train — apply same factors to test and holdout
    X_train, X_test, scale_factors = tanh_scale_train_apply_test(
        X_train_raw, X_test,
        scale_cols=SCALE_FEATURES, passthrough_cols=PASSTHROUGH_FEATURES
    )
    X_holdout = apply_scale_factors(X_holdout, scale_factors, PASSTHROUGH_FEATURES)

    logger.info("Scaling complete. Train bars after embargo purge: %d", len(X_train))

    # --- Evolution state ---
    seed_programs       = None
    generation_log      = []     # list of dicts per generation
    all_winners         = []     # formulas that survived TEST
    seen_hashes         = set()
    best_test_sharpe    = -np.inf
    patience_counter    = 0
    best_overall_model  = None
    best_overall_meta   = None

    for gen in range(1, MAX_GENERATIONS + 1):
        logger.info("\n%s", "─" * 60)
        logger.info("GENERATION %d / %d", gen, MAX_GENERATIONS)
        logger.info("─" * 60)

        try:
            gp_model    = train_gp_model(
                X_train, y_train_raw,
                seed_programs=seed_programs,
                fold=gen           # fold=gen gives per-generation random_state
            )
            formula_str = str(gp_model._program)
            program_len = len(gp_model._program.program) \
                          if hasattr(gp_model._program, 'program') else 0

        except Exception as exc:
            logger.error("GP training failed on gen %d: %s", gen, exc, exc_info=True)
            seed_programs = None
            continue

        # --- Degenerate formula guard ---
        if program_len > 80:
            logger.warning("Gen %d: formula too complex (len=%d) — skipping.", gen, program_len)
            seed_programs = None
            continue

        train_signals = gp_model.predict(X_train.values)
        entry_pct = np.percentile(train_signals, 90)
        exit_pct  = np.percentile(train_signals, 10)

        # --- Threshold collapse guard ---
        if entry_pct >= 1.0 and exit_pct <= 0.0:
            logger.warning("Gen %d: binary threshold collapse (Buy=%.2f, Sell=%.2f) — skipping.",
                           gen, entry_pct, exit_pct)
            seed_programs = None
            continue

        logger.info("Gen %d | Formula (len=%d): %s", gen, program_len, formula_str)
        logger.info("Gen %d | Thresholds → Buy: %.4f | Sell: %.4f",
                    gen, entry_pct, exit_pct)

        # --- OOS Evaluation on TEST ---
        portfolio, stats = evaluate_formula_with_vectorbt(
            gp_model, X_test, raw_test, entry_pct, exit_pct
        )
        total_return = float(stats.get('Total Return [%]', 0) or 0)
        sharpe       = float(stats.get('Sharpe Ratio',     0) or 0)
        max_dd       = float(stats.get('Max Drawdown [%]', 100) or 100)
        win_rate     = float(stats.get('Win Rate [%]',     0) or 0)

        logger.info("Gen %d | TEST → Return: %.2f%% | Sharpe: %.2f | DD: %.1f%% | WR: %.1f%%",
                    gen, total_return, sharpe, max_dd, win_rate)

        gen_record = {
            'generation':    gen,
            'formula':       formula_str,
            'formula_len':   program_len,
            'return_pct':    total_return,
            'sharpe':        sharpe,
            'max_dd':        max_dd,
            'win_rate':      win_rate,
            'buy_threshold': float(entry_pct),
            'sell_threshold':float(exit_pct),
            'survived':      False
        }

        # --- Survival criteria ---
        if (total_return > MIN_RETURN_TO_SURVIVE
                and sharpe   > MIN_SHARPE_TO_SURVIVE
                and max_dd   < MAX_DD_TO_SURVIVE):

            formula_hash = hash_formula(formula_str)
            if formula_hash in seen_hashes:
                logger.warning("Gen %d: duplicate formula — not logging, but seeding next gen.", gen)
            else:
                seen_hashes.add(formula_hash)
                gen_record['survived']     = True
                gen_record['formula_hash'] = formula_hash
                all_winners.append(gen_record)

                stats_out = stats.to_frame('value') if hasattr(stats, 'to_frame') \
                            else pd.DataFrame(stats)
                stats_out.to_csv(f"outputs/generations/gen_{gen:02d}_winner.csv")

                logger.info("Gen %d ✓ SURVIVOR — adding to hall of fame.", gen)

                # Track best model for holdout
                if sharpe > best_test_sharpe:
                    best_test_sharpe   = sharpe
                    best_overall_model = gp_model
                    best_overall_meta  = gen_record

            # Extract elite seeds regardless of dedup
            seed_programs = extract_elite_programs(gp_model)
            logger.info("Gen %d: extracted %d elite seeds → gen %d.",
                        gen, len(seed_programs), gen + 1)

            # Early stop patience reset on improvement
            if sharpe > best_test_sharpe:
                patience_counter = 0
            else:
                patience_counter += 1

        else:
            logger.info("Gen %d ✗ FAILED TEST — resetting seeds.", gen)
            seed_programs    = None
            patience_counter += 1

        generation_log.append(gen_record)

        # --- Early stopping ---
        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(
                "Early stopping triggered after %d consecutive non-improving generations.",
                patience_counter
            )
            break

        gc.collect()

    # -----------------------------------------------------------------------
    # FINAL HOLDOUT VALIDATION — touched exactly once
    # -----------------------------------------------------------------------
    logger.info("\n%s", "=" * 60)
    logger.info("HOLDOUT VALIDATION (touched once, never seen during evolution)")
    logger.info("=" * 60)

    if best_overall_model is not None:
        h_portfolio, h_stats = evaluate_formula_with_vectorbt(
            best_overall_model, X_holdout, raw_holdout,
            best_overall_meta['buy_threshold'],
            best_overall_meta['sell_threshold']
        )
        h_return  = float(h_stats.get('Total Return [%]', 0) or 0)
        h_sharpe  = float(h_stats.get('Sharpe Ratio',     0) or 0)
        h_dd      = float(h_stats.get('Max Drawdown [%]', 0) or 0)
        h_wr      = float(h_stats.get('Win Rate [%]',     0) or 0)

        logger.info("HOLDOUT RESULT | Return: %.2f%% | Sharpe: %.2f | DD: %.1f%% | WR: %.1f%%",
                    h_return, h_sharpe, h_dd, h_wr)
        logger.info("Formula: %s", best_overall_meta['formula'])
    else:
        logger.warning("No surviving formula to validate on holdout.")
        h_return = h_sharpe = h_dd = h_wr = 0

    # -----------------------------------------------------------------------
    # WRITE RESULTS
    # -----------------------------------------------------------------------
    _write_results(all_winners, generation_log, data_path,
                   best_overall_meta, h_return, h_sharpe, h_dd, h_wr)


def _write_results(all_winners, generation_log, data_path,
                   best_meta, h_return, h_sharpe, h_dd, h_wr):
    log_file = "outputs/winning_formulas.log"
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Run       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data      : {data_path}\n")
        f.write(f"Params    : ATR_MULT={ORACLE_ATR_MULT} | MAX_HOLD={ORACLE_MAX_HOLD}\n")
        f.write(f"Split     : Train={TRAIN_RATIO:.0%} | Test={TEST_RATIO:.0%} | Holdout={HOLDOUT_RATIO:.0%}\n")
        f.write(f"Survivors : {len(all_winners)} formulas across {len(generation_log)} generations\n")
        f.write("-" * 60 + "\n")

        for w in all_winners:
            f.write(f"Gen {w['generation']:>3} | Ret: {w['return_pct']:>8.2f}% | "
                    f"Sharpe: {w['sharpe']:.2f} | DD: {w['max_dd']:.1f}% | WR: {w['win_rate']:.1f}%\n")
            f.write(f"Hash      : {w.get('formula_hash','N/A')}\n")
            f.write(f"Thresholds: Buy>{w['buy_threshold']:.4f}  Sell<{w['sell_threshold']:.4f}\n")
            f.write(f"Logic     : {w['formula']}\n\n")

        if best_meta:
            f.write("─" * 60 + "\n")
            f.write("HOLDOUT VALIDATION (best formula by TEST Sharpe)\n")
            f.write(f"  Formula : {best_meta['formula']}\n")
            f.write(f"  Return  : {h_return:.2f}%\n")
            f.write(f"  Sharpe  : {h_sharpe:.2f}\n")
            f.write(f"  Max DD  : {h_dd:.1f}%\n")
            f.write(f"  Win Rate: {h_wr:.1f}%\n")

    # Generation evolution table
    gen_df = pd.DataFrame(generation_log)
    gen_df.to_csv("outputs/generations/evolution_log.csv", index=False)
    logger.info("Results written to %s", log_file)
    logger.info("Evolution log: outputs/generations/evolution_log.csv")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_directories()
    DATAPATH = "data/NIFTY 50_30minute 1(in).csv"

    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(f"Data file not found: '{DATAPATH}'")

    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        generational_evolution(
            df_raw, df_features, y_targets,
            data_path=DATAPATH
        )
    except Exception as exc:
        logger.critical("Pipeline crashed: %s", exc, exc_info=True)
