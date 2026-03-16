import copy
import hashlib
import logging
import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUSTOM LOGICAL & COMPARISON OPERATORS
# ---------------------------------------------------------------------------

def _gt(x1, x2):       return np.where(x1 > x2, 1.0, 0.0)
def _lt(x1, x2):       return np.where(x1 < x2, 1.0, 0.0)
def _eq(x1, x2):       return np.where(np.isclose(x1, x2, rtol=1e-05, atol=1e-08), 1.0, 0.0)
def _and(x1, x2):      return np.minimum(x1, x2)
def _or(x1, x2):       return np.maximum(x1, x2)
def _if_then(c, t, f): return np.where(c > 0.0, t, f)

greater_than = make_function(function=_gt,      name='gt',      arity=2)
less_than    = make_function(function=_lt,      name='lt',      arity=2)
equal_to     = make_function(function=_eq,      name='eq',      arity=2)
logical_and  = make_function(function=_and,     name='and',     arity=2)
logical_or   = make_function(function=_or,      name='or',      arity=2)
if_then      = make_function(function=_if_then, name='if_then', arity=3)

TRADING_FUNCTIONS = [
    'add', 'sub', 'mul', 'div', 'max', 'min', 'abs', 'neg',
    greater_than, less_than, equal_to, logical_and, logical_or, if_then
]

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
POPULATION_SIZE = 3000
SEED_FRACTION   = 0.20   # top 20% of prior fold seeded into next
MUTATION_BOOST  = 0.15   # elevated subtree-mutation for seeded individuals
                          # prevents gene lock-in / premature convergence


# ---------------------------------------------------------------------------
# POPULATION SEED UTILITIES
# ---------------------------------------------------------------------------

def hash_formula(program_str: str) -> str:
    """SHA-256 fingerprint of a formula string for deduplication."""
    return hashlib.sha256(program_str.encode()).hexdigest()


def extract_elite_programs(fitted_gp: SymbolicRegressor, top_n: int = None) -> list:
    """
    Extract deep-copies of the top-N fittest _Program objects from the
    last generation of a fitted SymbolicRegressor.

    Deep-copied to prevent shared-state mutation bugs between fold iterations.
    """
    if not hasattr(fitted_gp, '_programs') or not fitted_gp._programs:
        logger.warning("No _programs found on fitted model — returning empty seed list.")
        return []

    last_gen = fitted_gp._programs[-1]
    top_n    = top_n or int(SEED_FRACTION * POPULATION_SIZE)

    valid = [p for p in last_gen if p is not None and hasattr(p, 'fitness_')]
    elite = sorted(valid, key=lambda p: p.fitness_, reverse=True)[:top_n]

    logger.info("Extracted %d elite programs (pool size=%d).", len(elite), len(valid))
    return [copy.deepcopy(p) for p in elite]


# ---------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------------

def train_gp_model(
    X_train,
    y_train,
    seed_programs: list = None,
    fold: int = 0
) -> SymbolicRegressor:
    """
    Train a SymbolicRegressor with optional cross-fold warm-starting.

    Parameters
    ----------
    X_train       : pd.DataFrame — float32 feature matrix (fold-scaled)
    y_train       : pd.Series   — float32 oracle targets
    seed_programs : list[_Program] | None
                    Elite _Program objects from extract_elite_programs() on
                    the previous fold's fitted model. None = cold start.
    fold          : int — current fold index.
                    Used as random_state for per-fold population diversity.

    Injection Strategy
    ------------------
    gplearn does not expose a public warm-start population API.
    The production-safe workaround:
      Step A — run 1 generation to allocate _programs[0] structure.
      Step B — overwrite the weakest n_seeds slots with elite seeds.
      Step C — resume with warm_start=True for remaining 59 generations.

    This avoids forking gplearn while achieving true cross-fold gene carryover.
    """
    logger.info("[Fold %d] Initialising GP Engine...", fold)
    feature_names = list(X_train.columns)

    # FIX 1 & novelty pressure:
    # - random_state=fold   → different initial population per fold
    # - subtree_mut elevated → seeded individuals are mutated, not cloned
    subtree_mut = MUTATION_BOOST if seed_programs else 0.10

    est_gp = SymbolicRegressor(
        population_size      = POPULATION_SIZE,
        generations          = 60,
        tournament_size      = 100,
        p_crossover          = 0.6,
        p_subtree_mutation   = subtree_mut,   # elevated when seeding
        p_hoist_mutation     = 0.1,
        p_point_mutation     = 0.1,
        max_samples          = 0.7,
        parsimony_coefficient= 0.005,
        function_set         = TRADING_FUNCTIONS,
        init_depth           = (3, 6),
        metric               = 'pearson',
        feature_names        = feature_names,
        n_jobs               = 2,
        verbose              = 1,
        warm_start           = False,         # managed manually below
        random_state         = fold,          # FIX 1: per-fold diversity
    )

    if seed_programs:
        n_seeds = min(len(seed_programs), int(SEED_FRACTION * POPULATION_SIZE))
        logger.info("[Fold %d] Seeding %d elite programs from prior fold.", fold, n_seeds)

        # Step A: 1-generation bootstrap — allocates _programs[0]
        est_gp.generations = 1
        est_gp.fit(X_train.values, y_train.values)

        # Step B: replace weakest n_seeds slots with elite seeds
        gen0  = est_gp._programs[-1]   # list of _Program objects, len == pop_size
        valid = [(i, p) for i, p in enumerate(gen0)
                 if p is not None and hasattr(p, 'fitness_')]
        worst = sorted(valid, key=lambda t: t[1].fitness_)[:n_seeds]

        for slot, (pop_idx, _) in enumerate(worst):
            gen0[pop_idx] = copy.deepcopy(seed_programs[slot])

        logger.info("[Fold %d] Replaced %d weakest with elite seeds.", fold, len(worst))

        # Step C: resume remaining 59 generations
        est_gp.generations = 60
        est_gp.warm_start  = True
        est_gp.fit(X_train.values, y_train.values)

    else:
        logger.info("[Fold %d] No seeds — cold-start evolution.", fold)
        est_gp.fit(X_train.values, y_train.values)

    best = str(est_gp._program)
    logger.info("[Fold %d] Best formula: %s", fold, best)
    print(f"\n[Fold {fold}] Best Formula: {best}")
    return est_gp
