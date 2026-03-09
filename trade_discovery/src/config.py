# --- STRATEGY CONFIGURATION ---

# Target Generation / Backtest Params
# 52 bars at 30-minute resolution is approximately 4 trading sessions
# when using ~13 bars per day for Indian cash-session style data.
ORACLE_MAX_HOLD = 52

# ASYMMETRIC TRIPLE BARRIER MULTIPLIERS (ATR Units)
# These define the Take Profit and Stop Loss distances for BOTH
# target generation and out-of-sample evaluation.
TP_ATR_MULT = 4.0
SL_ATR_MULT = 1.5

# Execution Frictions
FEE_PER_SIDE = 0.0003
SLIPPAGE = 0.0001

# Absolute score floor required before any trade is allowed.
# Keep this unchanged for the first post-fix validation run, then retune
# from observed OOS coverage after the stricter evaluator is live.
ABSOLUTE_EDGE_FLOOR = 0.0080

# Feature Engineering Params
OB_ATR_MULT = 0.5

# GP Signal Calibration Thresholds
# These are percentile levels, not raw score values.
ENTRY_PCT = 90
EXIT_PCT = 10

# Optional for future multi-restart orchestration in the pipeline.
# Safe to keep here even if not yet consumed directly.
GP_RESTARTS = 3

# Fitness Function Constraints
MIN_LONG = 25
MIN_SHORT = 25
MIN_TRADES = 100

# Numeric Constraints & Regularization
EPS = 1e-8
STD_FLOOR = 1e-6
PF_SMOOTH_K = 1e-2
PF_MAX = 100.0
SHARPE_LAMBDA = 0.10
RETURN_LAMBDA = 2.0

# WFO / Data Params
TRAIN_MONTHS = 36
TEST_MONTHS = 6
DATAPATH = "data/NIFTY 50_30minute 1(in).csv"

# --- NEW PENALTY TERMS ---
IMBALANCE_LAMBDA = 2.0
ACTIVITY_FLOOR = 0.08
ACTIVITY_LAMBDA = 0.5
