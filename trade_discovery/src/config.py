# --- STRATEGY CONFIGURATION ---

# Oracle & Backtest Params
ORACLE_MAX_HOLD = 96      # Max hold time (e.g., 48 hours at 30m bars)
ORACLE_ATR_MULT = 3.5    # Dynamic Volatility Multiplier (Single Source of Truth)
FEE_PER_SIDE    = 0.0003  # 0.03% Transaction Fee
SLIPPAGE        = 0.0001  # 0.01% Slippage per side

# Feature Engineering Params
OB_ATR_MULT     = 0.5     # Width of Order Block zones in ATR units

# GP Engine Training Thresholds (Option B: Ranks)
ENTRY_PCT = 80            # Percentile for Long Signal Calibration
EXIT_PCT = 20             # Percentile for Short Signal Calibration

# Fitness Function Constraints
MIN_LONG = 3
MIN_SHORT = 3
MIN_TRADES = 12

# Numeric Constraints & Regularization
EPS = 1e-8
STD_FLOOR = 1e-6
PF_SMOOTH_K = 1e-2
PF_MAX = 500.0
SHARPE_LAMBDA = 0.05
RETURN_LAMBDA = 10.0

# WFO / Data Params
TRAIN_MONTHS    = 30
TEST_MONTHS     = 6
DATAPATH        = "data/Nifty-30min-4year(in).csv"
