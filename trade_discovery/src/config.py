# --- STRATEGY CONFIGURATION ---

# Target Generation / Backtest Params
# NOTE:
# ORACLE_ATR_MULT is retained for compatibility with the existing pipeline name,
# but it now represents the symmetric TBM barrier multiple.
ORACLE_MAX_HOLD = 96       # Max hold time (e.g., 48 hours at 30m bars)
ORACLE_ATR_MULT = 2.0      # Continuous TBM barrier multiple (+/- 2 ATR)

# Execution Frictions
FEE_PER_SIDE = 0.0003      # 0.03% transaction fee
SLIPPAGE = 0.0001          # 0.01% slippage per side

# Absolute score floor required before any trade is allowed.
# Round-trip friction = 2 * (fee + slippage) = 0.0008 (0.08%)
# Add a small safety margin so flat/choppy regimes stay flat.
ABSOLUTE_EDGE_FLOOR = 0.0010  # 0.10%

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
MIN_TRADES = 12

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
DATAPATH = "data/NIFTYNEXT50_30min_4Y.csv"
