import pandas as pd
import numpy as np
import os
import gc
import importlib
from typing import Tuple
  
# --- GLOBAL CONFIG ---
ORACLE_MAX_HOLD = 96     # 48 hours at 30m bars
ORACLE_ATR_MULT = 3.7   # Match this in 02_target_generator.py and 04_vectorbt_evaluator.py
FEE_PER_SIDE = 0.0003
SLIPPAGE = 0.0001

# Using importlib to handle modules starting with digits
fe = importlib.import_module("src.01_feature_engineering")
tg = importlib.import_module("src.02_target_generator")
gp = importlib.import_module("src.03_gp_engine")
vbteval = importlib.import_module("src.04_vectorbt_evaluator")

calculate_features = fe.calculate_features
generate_oracle_targets = tg.generate_oracle_targets
train_gp_model = gp.train_gp_model
evaluate_formula_with_vectorbt = vbteval.evaluate_formula_with_vectorbt

# Import explicitly defined feature contracts
PASSTHROUGH_FEATURES = fe.PASSTHROUGH_FEATURES
SCALE_FEATURES = fe.SCALE_FEATURES

def setup_directories():
    """Ensure output directories exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/vectorbt_stats", exist_ok=True)

def load_and_prepare_data(filepath):
    """Loads raw OHLC data and prepares features & targets."""
    print(f"Loading raw data from {filepath}...")
    df_raw = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
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

    df_features, y_targets = generate_oracle_targets(
        df_raw, df_features, 
        max_hold=ORACLE_MAX_HOLD, 
        atr_mult=ORACLE_ATR_MULT,
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE
    )

    df_raw = df_raw.loc[df_features.index]
    df_raw = df_raw.astype(np.float32)

    return df_raw, df_features, y_targets

def tanh_scale_train_apply_test(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    scale_cols: list, 
    passthrough_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Zero-Preserving Tanh Scaling to unbounded features.
    Prevents variance dominance in GP while preserving structural zeroes (e.g., indicator crosses).
    Enforces strict Train/Test isolation to prevent lookahead bias.
    """
    original_columns = X_train.columns
    actual_scale_cols = [c for c in scale_cols if c in X_train.columns]
    actual_pass_cols = [c for c in passthrough_cols if c in X_train.columns]
    
    X_train_scaled = pd.DataFrame(index=X_train.index)
    X_test_scaled = pd.DataFrame(index=X_test.index)
    
    # Epsilon constraint for Zero-Division Prevention in flat markets
    EPS = 1e-8
    
    if actual_scale_cols:
        train_scale_view = X_train[actual_scale_cols]
        test_scale_view = X_test[actual_scale_cols]
        
        # Fit on Train ONLY: 75th percentile of absolute values
        # Preserves 0.0 as the structural pivot point while scaling the bulk of data
        scale_factors = np.percentile(np.abs(train_scale_view), 75, axis=0)
        scale_factors = np.maximum(scale_factors, EPS)
        
        # Apply smooth monotonic Tanh compression
        X_train_scaled[actual_scale_cols] = np.tanh(train_scale_view / scale_factors).astype(np.float32)
        X_test_scaled[actual_scale_cols] = np.tanh(test_scale_view / scale_factors).astype(np.float32)
        
    if actual_pass_cols:
        # Passthrough features are already bounded in feature engineering
        X_train_scaled[actual_pass_cols] = X_train[actual_pass_cols].astype(np.float32)
        X_test_scaled[actual_pass_cols] = X_test[actual_pass_cols].astype(np.float32)
        
    return X_train_scaled[original_columns], X_test_scaled[original_columns]

def walk_forward_optimization(df_raw, df_features, y_targets, train_months=6, test_months=6, data_path="Unknown"):
    """Executes Walk-Forward Optimization."""
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
            print("Reached end of dataset. WFO Complete.")
            break

        print(f"\n--- FOLD {fold} ---")
        print(f"Train Window: {current_train_start.date()} to {train_end.date()}")
        print(f"Test Window: {train_end.date()} to {test_end.date()}")

        train_end_inclusive = train_end - pd.Timedelta(nanoseconds=1)
        
        X_train_raw = df_features.loc[current_train_start:train_end_inclusive]
        y_train_raw = y_targets.loc[current_train_start:train_end_inclusive]
        
        X_test = df_features.loc[train_end:test_end]
        raw_test = df_raw.loc[train_end:test_end]

        if len(X_train_raw) > ORACLE_MAX_HOLD:
            X_train = X_train_raw.iloc[:-ORACLE_MAX_HOLD]
            y_train = y_train_raw.iloc[:-ORACLE_MAX_HOLD]
        else:
            print(f"-> Fold {fold} Training set too short to purge. Skipping.")
            current_train_start += pd.DateOffset(months=test_months)
            continue

        if len(X_train) < 500 or len(X_test) < 200:
            print("Not enough data in this fold after purging. Skipping...")
            current_train_start += pd.DateOffset(months=test_months)
            continue

        # Fold-wise normalization with zero-preserving tanh scaling
        X_train, X_test = tanh_scale_train_apply_test(
            X_train, X_test, 
            scale_cols=SCALE_FEATURES, 
            passthrough_cols=PASSTHROUGH_FEATURES
        )

        try:
            # 1. Train GP
            gp_model = train_gp_model(X_train, y_train)
            formula_str = str(gp_model._program)

            train_signals = gp_model.predict(X_train.values)
            entry_pct = np.percentile(train_signals, 90)
            exit_pct = np.percentile(train_signals, 10)
            
            print(f"-> Training Thresholds | Buy: {entry_pct:.4f}, Sell: {exit_pct:.4f}")

        except Exception as e:
            print(f"GP Training failed on Fold {fold}: {e}")
            break

        # 2. Evaluate OOS
        portfolio, stats = evaluate_formula_with_vectorbt(
            gp_model, X_test, raw_test, entry_pct, exit_pct,
            fees=FEE_PER_SIDE, slippage=SLIPPAGE
        )

        total_return = stats.get('Total Return [%]', 0)
        sharpe = stats.get('Sharpe Ratio', 0)

        if pd.isna(total_return): total_return = 0
        if pd.isna(sharpe): sharpe = 0

        # 3. Survival Criteria
        if total_return > 20 and sharpe > 0.5:
            print(f"-> SUCCESS! Formula survived OOS. Return: {total_return:.2f}%, Sharpe: {sharpe:.2f}")
            winning_formulas.append({
                'fold': fold,
                'formula': formula_str,
                'return_pct': float(total_return),
                'sharpe': float(sharpe),
                'buy_threshold': float(entry_pct),
                'sell_threshold': float(exit_pct)
            })

            if hasattr(stats, 'to_frame'):
                stats.to_frame(name='value').to_csv(f"outputs/vectorbt_stats/fold_{fold}_winner.csv")
            else:
                pd.DataFrame(stats).to_csv(f"outputs/vectorbt_stats/fold_{fold}_winner.csv")
        else:
            print(f"-> FAILED. Formula collapsed in OOS. Return: {total_return:.2f}%, Sharpe: {sharpe:.2f}")
            print("Discarding formula and moving to next fold.")

        current_train_start += pd.DateOffset(months=test_months)
        fold += 1

        if 'X_train' in locals(): del X_train
        if 'y_train' in locals(): del y_train
        if 'X_test' in locals(): del X_test
        if 'raw_test' in locals(): del raw_test
        if 'gp_model' in locals(): del gp_model
        if 'portfolio' in locals(): del portfolio
        gc.collect()

    print("=" * 50)
    print("DISCOVERY RUN COMPLETE")
    print("=" * 50)

    if winning_formulas:
        print(f"Found {len(winning_formulas)} robust strategies.")
        log_file = "outputs/winning_formulas.log"
        with open(log_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Discovery Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {data_path}\n")
            f.write(f"Parameters: ATR_MULT={ORACLE_ATR_MULT}, MAX_HOLD={ORACLE_MAX_HOLD}\n")
            f.write("-" * 60 + "\n")
            for w in winning_formulas:
                f.write(f"Fold {w['fold']} | Ret: {w['return_pct']:.2f}% | Sharpe: {w['sharpe']:.2f}\n")
                f.write(f"Thresholds: Buy>{w['buy_threshold']:.4f}, Sell<{w['sell_threshold']:.4f}\n")
                f.write(f"Logic: {w['formula']}\n\n")
        print("Winners appended to logfile.")
    else:
        print("No robust strategies found. Consider adjusting parameters or providing more data.")
if __name__ == "__main__":
    setup_directories()
    # Fixed to match actual filename on disk as per previous conversation context
    DATAPATH = "data/Nifty-30min-4year(in).csv"
    

    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(f"CRITICAL ERROR: Data file not found at {DATAPATH}. Please check your filename and directory.")

    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        walk_forward_optimization(df_raw, df_features, y_targets, train_months=30, test_months=6, data_path=DATAPATH)
    except Exception as e:
        print(f"Pipeline crashed: {e}")

# auto-commit test
