import numpy as np
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor

# --- CUSTOM LOGICAL & COMPARISON OPERATORS ---
def _gt(x1, x2):
    return np.where(x1 > x2, 1.0, 0.0)

def _lt(x1, x2):
    return np.where(x1 < x2, 1.0, 0.0)

def _eq(x1, x2):
    return np.where(np.isclose(x1, x2, rtol=1e-05, atol=1e-08), 1.0, 0.0)

def _and(x1, x2):
    return np.minimum(x1, x2)

def _or(x1, x2):
    return np.maximum(x1, x2)

def _if_then(condition, out_true, out_false):
    return np.where(condition > 0.0, out_true, out_false)

greater_than = make_function(function=_gt, name='gt', arity=2)
less_than    = make_function(function=_lt, name='lt', arity=2)
equal_to     = make_function(function=_eq, name='eq', arity=2)
logical_and  = make_function(function=_and, name='and', arity=2)
logical_or   = make_function(function=_or,  name='or',  arity=2)
if_then      = make_function(function=_if_then, name='if_then', arity=3)

trading_functions = [
    'add', 'sub', 'mul', 'div', 'max', 'min', 'abs', 'neg',
    greater_than, less_than, equal_to, logical_and, logical_or, if_then
]

# ─────────────────────────────────────────────────────────────────────────
# TRADING FITNESS FUNCTION
# Replaces Pearson. Judges each formula by how well it TRADES against
# the triple-barrier labels, not just how well it correlates with them.
#
# y      = triple barrier labels (+1 long win, -1 short win, 0 timeout)
# y_pred = raw continuous output of the GP formula
# w      = sample weights (required by gplearn API, not used here)
#
# Fitness = Sharpe × WinRate / max(|MaxDD| / 10, 1)
# ─────────────────────────────────────────────────────────────────────────
def _trading_fitness(y, y_pred, w):
    # Threshold: top 30% = long signal, bottom 30% = short signal
    long_thresh  = np.percentile(y_pred, 70)
    short_thresh = np.percentile(y_pred, 30)

    long_mask  = y_pred >= long_thresh
    short_mask = y_pred <= short_thresh

    # Simulate P&L using triple barrier label as outcome:
    # Long trade on +1 label  → profit (+1), on -1 label → loss (-1)
    # Short trade on -1 label → profit (+1), on +1 label → loss (-1)
    # No trade (0 zone)       → 0
    raw_returns = np.where(long_mask,  y,
                  np.where(short_mask, -y,
                  0.0))

    n_trades = np.sum(long_mask | short_mask)

    # Need at least 20 trades to compute meaningful stats
    if n_trades < 20:
        return 0.0

    trade_returns = raw_returns[long_mask | short_mask]

    mean_r  = trade_returns.mean()
    std_r   = trade_returns.std() + 1e-9
    sharpe  = mean_r / std_r  # annualisation not needed — relative comparison

    win_rate = (trade_returns > 0).mean()  # fraction, 0.0 to 1.0

    # Max drawdown on cumulative P&L curve
    cum     = np.cumsum(trade_returns)
    max_dd  = (np.maximum.accumulate(cum) - cum).max() + 1e-9

    fitness = sharpe * win_rate / max(abs(max_dd) / 10.0, 1.0)

    # gplearn minimises by default when greater_is_better=False,
    # so return negative (we want to MAXIMISE fitness)
    return -fitness  # negative because gplearn minimises


trading_metric = make_fitness(
    function=_trading_fitness,
    greater_is_better=False   # gplearn will minimise → minimising -fitness = maximising fitness
)

# ─────────────────────────────────────────────────────────────────────────

def train_gp_model(X_train, y_train, random_state=42):
    """
    X_train: float32 DataFrame of features
    y_train: float32 Series of triple-barrier Oracle targets (+1, -1, 0)
    """
    print("Initializing GP Engine with Trading Fitness...")

    feature_names = list(X_train.columns)

    est_gp = SymbolicRegressor(
        population_size=3000,
        generations=60,
        tournament_size=100,
        p_crossover=0.6,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.1,
        p_point_mutation=0.1,
        max_samples=0.7,
        parsimony_coefficient=0.005,
        function_set=trading_functions,
        init_depth=(3, 6),
        metric=trading_metric,       # ← ONLY MEANINGFUL CHANGE FROM BEFORE
        feature_names=feature_names,
        n_jobs=2,
        verbose=1,
        random_state=random_state
    )

    print("Starting Evolution...")
    est_gp.fit(X_train.values, y_train.values)
    print("\nBest Formula Found:")
    print(est_gp._program)
    return est_gp
