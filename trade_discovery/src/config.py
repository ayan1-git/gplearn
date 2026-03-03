# --- STRATEGY CONFIGURATION ---

# Oracle & Backtest Params
ORACLE_MAX_HOLD = 96      # Max hold time (e.g., 48 hours at 30m bars)
ORACLE_ATR_MULT = 1.5     # Dynamic Volatility Multiplier (Single Source of Truth)
FEE_PER_SIDE    = 0.0003  # 0.03% Transaction Fee
SLIPPAGE        = 0.0001  # 0.01% Slippage per side

# Feature Engineering Params
OB_ATR_MULT     = 0.5     # Width of Order Block zones in ATR units

# WFO / Data Params
TRAIN_MONTHS    = 30
TEST_MONTHS     = 6
DATAPATH        = "data/NIFTYNEXT50_30min_4Y.csv"
