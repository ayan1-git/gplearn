import importlib

import numpy as np
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

# ---------------------------------------------------------------------
# CUSTOM LOGICAL & COMPARISON OPERATORS
# ---------------------------------------------------------------------


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


def _bounded_div(x1, x2):
    eps = 1e-3
    clip = 5.0

    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)

    safe_denom = np.where(np.abs(x2) < eps, np.sign(x2) * eps, x2)
    safe_denom = np.where(safe_denom == 0.0, eps, safe_denom)

    out = x1 / safe_denom
    out = np.clip(out, -clip, clip)

    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


greater_than = make_function(function=_gt, name="gt", arity=2)
less_than = make_function(function=_lt, name="lt", arity=2)
equal_to = make_function(function=_eq, name="eq", arity=2)
logical_and = make_function(function=_and, name="and", arity=2)
logical_or = make_function(function=_or, name="or", arity=2)
if_then = make_function(function=_if_then, name="if_then", arity=3)
bounded_division = make_function(function=_bounded_div, name="div", arity=2)

trading_functions = [
    "add",
    "sub",
    "mul",
    bounded_division,   # replaces built-in "div"
    "max",
    "min",
    "abs",
    "neg",
    greater_than,
    less_than,
    equal_to,
    logical_and,
    logical_or,
    if_then,
]

# ---------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------

try:
    config = importlib.import_module("src.config")
    ENTRY_PCT = float(config.ENTRY_PCT)
    EXIT_PCT = float(config.EXIT_PCT)
    MIN_LONG = int(config.MIN_LONG)
    MIN_SHORT = int(config.MIN_SHORT)
    MIN_TRADES = int(config.MIN_TRADES)
    EPS = float(config.EPS)
    STD_FLOOR = float(config.STD_FLOOR)
    PF_SMOOTH_K = float(config.PF_SMOOTH_K)
    PF_MAX = float(config.PF_MAX)
    SHARPE_LAMBDA = float(config.SHARPE_LAMBDA)
    RETURN_LAMBDA = float(config.RETURN_LAMBDA)

    IMBALANCE_LAMBDA = float(getattr(config, "IMBALANCE_LAMBDA", 2.0))
    ACTIVITY_FLOOR = float(getattr(config, "ACTIVITY_FLOOR", 0.08))
    ACTIVITY_LAMBDA = float(getattr(config, "ACTIVITY_LAMBDA", 0.5))
    ABSOLUTE_EDGE_FLOOR = float(config.ABSOLUTE_EDGE_FLOOR)
except (ImportError, AttributeError):
    ENTRY_PCT, EXIT_PCT = 80.0, 20.0
    MIN_LONG, MIN_SHORT, MIN_TRADES = 3, 3, 12
    EPS, STD_FLOOR, PF_SMOOTH_K, PF_MAX = 1e-8, 1e-6, 1e-2, 500.0
    SHARPE_LAMBDA, RETURN_LAMBDA = 0.05, 10.0
    ABSOLUTE_EDGE_FLOOR = 0.0010

    IMBALANCE_LAMBDA = 2.0
    ACTIVITY_FLOOR = 0.08
    ACTIVITY_LAMBDA = 0.5


def _validate_training_inputs(X_train, y_train):
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must not be None.")

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train and y_train must be non-empty.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    if hasattr(X_train, "index") and hasattr(y_train, "index"):
        if not X_train.index.equals(y_train.index):
            raise ValueError("X_train and y_train indices must be aligned.")

    X_values = np.asarray(X_train.values, dtype=np.float32)
    y_values = np.asarray(y_train.values, dtype=np.float32)

    if X_values.ndim != 2:
        raise ValueError("X_train must be a 2D feature matrix.")
    if y_values.ndim != 1:
        raise ValueError("y_train must be a 1D target vector.")

    if not np.isfinite(X_values).all():
        raise ValueError("X_train contains non-finite values.")
    if not np.isfinite(y_values).all():
        raise ValueError("y_train contains non-finite values.")

    unique_targets = np.unique(y_values)
    if unique_targets.size < 2:
        raise ValueError("y_train must contain at least 2 unique target values.")

    return X_values, y_values


def _safe_percentile_thresholds(y_pred):
    buy = np.percentile(y_pred, ENTRY_PCT)
    sell = np.percentile(y_pred, EXIT_PCT)

    if not np.isfinite(buy) or not np.isfinite(sell):
        return None, None

    if sell >= buy:
        return None, None

    return float(buy), float(sell)


def _pf_sharpe_fitness(y, y_pred, w):
    y = np.asarray(y, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y.size == 0 or y_pred.size == 0:
        return 0.0
    if y.shape[0] != y_pred.shape[0]:
        return 0.0
    if not np.isfinite(y).all() or not np.isfinite(y_pred).all():
        return 0.0

    buy, sell = _safe_percentile_thresholds(y_pred)
    if buy is None or sell is None:
        return 0.0

    signal = np.zeros_like(y_pred, dtype=np.float32)

    long_rank_mask = y_pred > buy
    short_rank_mask = y_pred < sell

    long_edge_mask = y_pred >= ABSOLUTE_EDGE_FLOOR
    short_edge_mask = y_pred <= -ABSOLUTE_EDGE_FLOOR

    long_mask = long_rank_mask & long_edge_mask
    short_mask = short_rank_mask & short_edge_mask

    overlap = long_mask & short_mask
    if overlap.any():
        long_mask = long_mask & (~overlap)
        short_mask = short_mask & (~overlap)

    signal[long_mask] = 1.0
    signal[short_mask] = -1.0

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    n_trades = n_long + n_short

    imbalance_ratio = abs(n_long - n_short) / max(n_trades, 1)
    if imbalance_ratio > 0.75:
        return 0.0

    if (n_long < MIN_LONG) or (n_short < MIN_SHORT) or (n_trades < MIN_TRADES):
        return 0.0

    captured = signal * y

    trade_mask = signal != 0.0
    if not trade_mask.any():
        return 0.0

    realized = captured[trade_mask]
    if realized.size < MIN_TRADES:
        return 0.0
    if not np.isfinite(realized).all():
        return 0.0

    std = float(np.std(realized))
    if std < STD_FLOOR:
        return 0.0

    pos = realized[realized > 0.0]
    neg = realized[realized < 0.0]

    gross_profit = float(pos.sum()) if pos.size else 0.0
    gross_loss = float((-neg).sum()) if neg.size else 0.0

    pf = (gross_profit + PF_SMOOTH_K) / (gross_loss + PF_SMOOTH_K)
    pf = min(max(pf, 1e-6), PF_MAX)

    mean_realized = float(np.mean(realized))
    sharpe = mean_realized / (std + EPS)
    tot_ret = float(realized.sum())

    activity_ratio = realized.size / max(y.size, 1)
    if activity_ratio <= 0.0:
        return 0.0

    activity_penalty = 0.0
    if activity_ratio < ACTIVITY_FLOOR:
        activity_penalty = ACTIVITY_LAMBDA * (ACTIVITY_FLOOR - activity_ratio)

    score = (
        np.log(pf)
        + (SHARPE_LAMBDA * sharpe)
        + (RETURN_LAMBDA * tot_ret)
        - activity_penalty
        - (IMBALANCE_LAMBDA * imbalance_ratio)
    )

    return float(score)


pf_sharpe_metric = make_fitness(function=_pf_sharpe_fitness, greater_is_better=True)


def train_gp_model(X_train, y_train, random_state=42):
    """
    X_train: float32 DataFrame of features
    y_train: float32 Series of strict first-touch targets
    """
    X_values, y_values = _validate_training_inputs(X_train, y_train)
    feature_names = list(X_train.columns)

    est_gp = SymbolicRegressor(
        population_size=2500,
        generations=25,
        tournament_size=20,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.7,
        stopping_criteria=100.0,
        parsimony_coefficient=0.02,
        function_set=trading_functions,
        init_depth=(1, 3),
        init_method="half and half",
        metric=pf_sharpe_metric,
        feature_names=feature_names,
        n_jobs=2,
        verbose=0,
        random_state=random_state,
    )

    est_gp.fit(X_values, y_values)
    return est_gp
