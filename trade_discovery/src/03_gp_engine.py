import copy
import hashlib
import logging
import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

try:
    import src.config as config
    POPULATION_SIZE = getattr(config, "GP_POPULATION_SIZE", 3000)
    SEED_FRACTION   = getattr(config, "GP_SEED_FRACTION", 0.20)
    MUTATION_BOOST  = getattr(config, "GP_MUTATION_BOOST", 0.15)
    GENERATIONS     = getattr(config, "GP_GENERATIONS", 60)
    TOURNAMENT_SIZE = getattr(config, "GP_TOURNAMENT_SIZE", 100)
except ImportError:
    POPULATION_SIZE = 3000
    SEED_FRACTION   = 0.15
    MUTATION_BOOST  = 0.15
    GENERATIONS     = 60
    TOURNAMENT_SIZE = 100

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
# CONSTANTS (Imported from config or defaulted)
# ---------------------------------------------------------------------------


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
        generations          = GENERATIONS,
        tournament_size      = TOURNAMENT_SIZE,
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
        logger.info("[Fold %d] Will inject %d seeds via _programs pre-population.", fold, n_seeds)

        # Reduce parsimony during seeded folds to prevent depth-1 collapse
        # Seeds are already short; parsimony will kill complexity in 1 gen otherwise
        est_gp.parsimony_coefficient = 0.001   # ← reduced from 0.005

        # Run the full generations cold — but intercept after gen 0
        # using a single-generation pre-run, inject, then resume
        est_gp.generations = 1
        est_gp.warm_start  = False
        est_gp.fit(X_train.values, y_train.values)

        gen0        = est_gp._programs[-1]
        rng         = np.random.RandomState(fold + 1000)
        n_features  = X_train.shape[1]

        valid_gen0  = [(i, p) for i, p in enumerate(gen0)
                       if p is not None and hasattr(p, 'fitness_')]
        worst_slots = sorted(valid_gen0, key=lambda t: t[1].fitness_)[:n_seeds]

        for slot_rank, (pop_idx, _) in enumerate(worst_slots):
            seed = copy.deepcopy(seed_programs[slot_rank % len(seed_programs)])
            # Mutate terminals to restore variance
            if hasattr(seed, 'program') and len(seed.program) > 2:
                n_mutate = max(1, int(0.20 * len(seed.program)))
                for _ in range(n_mutate):
                    idx  = rng.randint(0, len(seed.program))
                    node = seed.program[idx]
                    if isinstance(node, int):
                        seed.program[idx] = rng.randint(0, n_features)
            gen0[pop_idx] = seed

        # ── KEY FIX: force population depth diversity before resuming ──
        # Rebuild avg_length signal by ensuring 30% of remaining pop has
        # depth >= 4 (prevents parsimony-driven collapse to depth-1 in gen 1)
        random_slots = [i for i, p in enumerate(gen0)
                        if p is not None and hasattr(p, 'program')
                        and len(getattr(p, 'program', [])) <= 3
                        and i not in {idx for idx, _ in worst_slots}]
        # Leave them as-is — they came from the random cold-start gen0 which
        # already has avg_length ~28. The issue is parsimony killing them in gen1.
        # Solution: reduce parsimony_coefficient for seeded folds (done above).

        est_gp.generations = GENERATIONS
        est_gp.warm_start  = True
        est_gp.fit(X_train.values, y_train.values)

    else:
        logger.info("[Fold %d] No seeds — cold-start evolution.", fold)
        est_gp.fit(X_train.values, y_train.values)

    best = str(est_gp._program)
    logger.info("[Fold %d] Best formula: %s", fold, best)
    print(f"\n[Fold {fold}] Best Formula: {best}")
    return est_gp
