# --- STRATEGY CONFIGURATION ---

# Target Generation / Backtest Params
ORACLE_MAX_HOLD = 52       # Max hold time (e.g., 48 hours at 30m bars)

# ASYMMETRIC TRIPLE BARRIER MULTIPLIERS (ATR Units)
# These define the Take Profit and Stop Loss distances for BOTH
# target generation (GP labels) and the VectorBT backtest.
TP_ATR_MULT = 3.0          # Target Profit: e.g., 3.0 ATR
SL_ATR_MULT = 1.5         # Stop Loss: e.g., 1.5 ATR (Reward/Risk = 2.0)

# Execution Frictions
FEE_PER_SIDE = 0.0003      # 0.03% transaction fee
SLIPPAGE = 0.0001          # 0.01% slippage per side

# Absolute score floor required before any trade is allowed.
# Round-trip friction = 2 * (fee + slippage) = 0.0008 (0.08%)
ABSOLUTE_EDGE_FLOOR = 0.0040  # 0.40% Selective threshold

# Feature Engineering Params
OB_ATR_MULT = 0.5         # Width of Order Block zones in ATR units

# GP Signal Calibration Thresholds
# These are percentile LEVELS, not raw score values.
ENTRY_PCT = 80            # Long when score is in the top 20% of recent history
EXIT_PCT = 20             # Short when score is in the bottom 20% of recent history
GP_RESTARTS = 3           # Number of GP runs per fold

# Fitness Function Constraints
MIN_LONG = 3
MIN_SHORT = 3
MIN_TRADES = 48

# Numeric Constraints & Regularization
EPS = 1e-8
STD_FLOOR = 1e-6
PF_SMOOTH_K = 1e-2
PF_MAX = 500.0
SHARPE_LAMBDA = 0.05
RETURN_LAMBDA = 10.0

# WFO / Data Params
TRAIN_MONTHS = 30
TEST_MONTHS = 6
DATAPATH = "data/Nifty-30min-4year(in).csv"
