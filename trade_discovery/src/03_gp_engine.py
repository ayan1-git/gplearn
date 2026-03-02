import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

# --- CUSTOM LOGICAL & COMPARISON OPERATORS ---
# Return continuous approximations or relative comparisons 
# instead of hard thresholding against 0.5

def _gt(x1, x2):
    return np.where(x1 > x2, 1.0, 0.0)

def _lt(x1, x2):
    return np.where(x1 < x2, 1.0, 0.0)

def _eq(x1, x2):
    # relative tolerance rather than absolute to handle arbitrary scales
    return np.where(np.isclose(x1, x2, rtol=1e-05, atol=1e-08), 1.0, 0.0)

def _and(x1, x2):
    # Continuous proxy for AND: min or multiplication of positive signals
    return np.minimum(x1, x2)

def _or(x1, x2):
    # Continuous proxy for OR: max
    return np.maximum(x1, x2)

def _if_then(condition, out_true, out_false):
    # Condition relies on whether condition > 0, which provides a relative 
    # zero-crossing boundary rather than an arbitrary 0.5
    return np.where(condition > 0.0, out_true, out_false)

# Registering functions for gplearn
greater_than = make_function(function=_gt, name='gt', arity=2)
less_than = make_function(function=_lt, name='lt', arity=2)
equal_to = make_function(function=_eq, name='eq', arity=2)
logical_and = make_function(function=_and, name='and', arity=2)
logical_or = make_function(function=_or, name='or', arity=2)
if_then = make_function(function=_if_then, name='if_then', arity=3)

# Build the function set
trading_functions = [
    'add', 'sub', 'mul', 'div', 'max', 'min', 'abs', 'neg',
    greater_than, less_than, equal_to, logical_and, logical_or, if_then
]

# ---------------------------------------------------------------------
# PF-dominant composite fitness (sniper-friendly, flat=neutral)
# ---------------------------------------------------------------------
ENTRY_PCT = 70
EXIT_PCT = 30
MIN_LONG = 3
MIN_SHORT = 3
MIN_TRADES = 12

EPS = 1e-8
STD_FLOOR = 1e-6
PF_SMOOTH_K = 1e-2
PF_MAX = 500.0
SHARPE_LAMBDA = 0.05
RETURN_LAMBDA = 10.0  # Encourage magnitude of captured returns

def _pf_sharpe_fitness(y, y_pred, w):
    y = np.asarray(y, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    buy = np.percentile(y_pred, ENTRY_PCT)
    sell = np.percentile(y_pred, EXIT_PCT)

    signal = np.zeros_like(y_pred, dtype=np.float32)
    long_mask = y_pred > buy
    short_mask = y_pred < sell
    signal[long_mask] = 1.0
    signal[short_mask] = -1.0

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    if (n_long < MIN_LONG) or (n_short < MIN_SHORT) or ((n_long + n_short) < MIN_TRADES):
        return 0.0

    captured = signal * y
    std = float(np.std(captured))
    if std < STD_FLOOR:
        return 0.0

    pos = captured[captured > 0]
    neg = captured[captured < 0]
    gross_profit = float(pos.sum()) if pos.size else 0.0
    gross_loss = float((-neg).sum()) if neg.size else 0.0

    pf = (gross_profit + PF_SMOOTH_K) / (gross_loss + PF_SMOOTH_K)
    pf = min(max(pf, 1e-6), PF_MAX)

    mean = float(np.mean(captured))
    sharpe = mean / (std + EPS)
    
    # Raw cumulative return proxy
    tot_ret = float(captured.sum())

    # Multi-objective: Log(PF) for quality, Sharpe for risk, tot_ret for magnitude
    return float(np.log(pf) + (SHARPE_LAMBDA * sharpe) + (RETURN_LAMBDA * tot_ret))

pf_sharpe_metric = make_fitness(function=_pf_sharpe_fitness, greater_is_better=True)

def train_gp_model(X_train, y_train):
    """
    X_train: float32 DataFrame of features
    y_train: float32 Series of Oracle targets
    """
    print("Initializing GP Engine (8GB RAM Safe Mode)...")
    
    # Get feature names from the DataFrame columns
    feature_names = list(X_train.columns)
    
    est_gp = SymbolicRegressor(
        population_size=5000,
        generations=40,
        tournament_size=50,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.7,             # Reduced to prevent overfitting and massive programs
        stopping_criteria=100.0,
        parsimony_coefficient=0.09,   # Significantly increased to penalize length
        function_set=trading_functions,
        init_depth=(1, 4),            # Scaled back initial complexity
        metric=pf_sharpe_metric,
        feature_names=feature_names,
        n_jobs=2,
        verbose=1,
        random_state=42
    )
    
    print("Starting Evolution...")
    est_gp.fit(X_train.values, y_train.values)
    print("\nBest Formula Found:")
    print(est_gp._program)
    return est_gp
