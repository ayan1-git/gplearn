from __future__ import annotations
import gc
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from collections import deque

import numpy as np
import pandas as pd
import logging
import importlib
try:
    config = importlib.import_module("src.config")
    OB_ATR_MULT = config.OB_ATR_MULT
except (ImportError, AttributeError):
    OB_ATR_MULT = 0.5  # Safe fallback for standalone testing

logger = logging.getLogger(__name__)

EPS = 1e-12

# --- FEATURE CONTRACTS FOR WFO SCALING ---
# Defined explicitly so the pipeline knows how to treat each feature.

PASSTHROUGH_FEATURES = [
    "feat_ob_supp_active", "feat_ob_res_active",    # Binary (0/1)
    "feat_session_sin", "feat_session_cos",         # Cyclical [-1, 1]
    "feat_icp", "feat_efficiency",                  # Construction-bounded [-1, 1]
    "feat_ob_supp_touches", "feat_ob_res_touches",  # Clipped [0, 1]
    "feat_momentum_rsi",                            # Bounded [-1, 1]
    "feat_momentum_stoch", "feat_trend_adx"          # NEW: Stochastic & ADX [-1, 1]
]

SCALE_FEATURES = [
    "feat_volatility_regime", "feat_dist_skew",
    "feat_zscore", "feat_momentum_mds", "feat_vol_asymmetry",
    "feat_ob_dist_supp", "feat_ob_dist_res",
    "feat_ichimoku_dist_tenkan", "feat_ichimoku_dist_kijun",
    "feat_ichimoku_dist_span_a", "feat_ichimoku_dist_span_b"
]

def clip_scale(series: pd.Series, bound: float = 1.0) -> pd.Series:
    """Hard clip to [-bound, bound]."""
    if bound <= 0:
        raise ValueError(f"clip_scale bound must be > 0")
    return series.clip(-bound, bound)

@dataclass(frozen=True)
class SessionConfig:
    open_time: str = "09:15"
    close_time: str = "15:30"
    tz: str = "Asia/Kolkata"

def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid hhmm time string: {hhmm}")
    h = int(parts[0])
    m = int(parts[1])
    if not (0 <= h <= 23) and (0 <= m <= 59):
        raise ValueError(f"Invalid time: {hhmm}")
    return h, m

def session_cyclic_position(
    index: pd.DatetimeIndex,
    session: SessionConfig,
    clip_outside_session: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Continuous time-of-day encoding using sin/cos."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex for session features")
    
    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert(session.tz)
        
    open_h, open_m = parse_hhmm(session.open_time)
    close_h, close_m = parse_hhmm(session.close_time)
    
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    session_len = close_minutes - open_minutes
    if session_len <= 0:
         raise ValueError("Session close must be after session open")
         
    minutes = idx.hour * 60 + idx.minute.astype(np.int32)
    
    if clip_outside_session:
        minutes = np.clip(minutes, open_minutes, close_minutes)
        minutes = pd.Series(minutes, index=index, name="minutes")
    else:
        minutes = pd.Series(minutes, index=index, name="minutes")
        minutes = minutes.where((minutes >= open_minutes) & (minutes <= close_minutes), np.nan)
        
    pos = (minutes - open_minutes) / float(session_len)
    angle = 2.0 * np.pi * pos
    
    feat_sin = np.sin(angle).astype(np.float32).rename("feat_session_sin")
    feat_cos = np.cos(angle).astype(np.float32).rename("feat_session_cos")
    return feat_sin, feat_cos

def momentum_divergence_score(
    log_ret: pd.Series, 
    fast_window: int, 
    slow_window: int, 
) -> pd.Series:
    """Multi-horizon momentum divergence."""
    if fast_window < 1 or slow_window < 2:
        raise ValueError("fast_window must be >= 1 and slow_window must be >= 2")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")
        
    fast_sum = log_ret.rolling(fast_window, min_periods=fast_window).sum()
    slow_sum = log_ret.rolling(slow_window, min_periods=slow_window).sum()
    slow_vol = log_ret.rolling(slow_window, min_periods=slow_window).std() + EPS
    
    mds = (fast_sum - slow_sum) / slow_vol
    return mds.rename("feat_momentum_mds")

def directional_vol_asymmetry(
    log_ret: pd.Series, 
    window: int, 
) -> pd.Series:
    """Directional volatility asymmetry."""
    if window < 2:
        raise ValueError("window must be >= 2")
        
    up = log_ret.clip(lower=0.0)
    down = log_ret.clip(upper=0.0)
    
    up_vol = up.rolling(window, min_periods=window).std()
    down_vol = down.rolling(window, min_periods=window).std()
    
    raw = (up_vol - down_vol) / (up_vol + down_vol + EPS)
    return raw.rename("feat_vol_asymmetry")

class OptimizedOrderBlockEngine:
    def __init__(
        self,
        internal_lookback: int = 5,
        swing_lookback: int = 20,
        atr_multiplier: float = 0.5,
        max_obs: int = 5,
        iou_threshold: float = 0.85,
        missing_value_fill: float = 5.0, 
    ):
        if internal_lookback >= swing_lookback:
            raise ValueError("internal_lookback must be strictly less than swing_lookback.")
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0, 1].")
            
        self.int_lb = internal_lookback
        self.swg_lb = swing_lookback
        self.atr_mult = atr_multiplier
        self.max_obs = max_obs
        self.iou_threshold = iou_threshold
        self.missing_fill = missing_value_fill
        self.max_window = swing_lookback * 2 + 1

    def get_pivot_flags(self, series: pd.Series, lookback: int, is_high: bool) -> np.ndarray:
        shifted = series.shift(lookback)
        left_bar = series.shift(lookback + 1)
        right_bar = series.shift(lookback - 1)
        
        if is_high:
            flags = (shifted > left_bar) & (shifted > right_bar)
        else:
            flags = (shifted < left_bar) & (shifted < right_bar)
            
        return flags.to_numpy()

    def iou_1d(self, bot_a: float, top_a: float, bot_b: float, top_b: float) -> float:
        intersect_top = min(top_a, top_b)
        intersect_bot = max(bot_a, bot_b)
        if intersect_top <= intersect_bot:
            return 0.0
        overlap = intersect_top - intersect_bot
        union = max(top_a, top_b) - min(bot_a, bot_b)
        return overlap / union if union > 0 else 0.0

    def is_duplicate_spatial(self, new_ob: Dict, queue: deque) -> bool:
        for ob in queue:
            if self.iou_1d(new_ob['bot'], new_ob['top'], ob['bot'], ob['top']) > self.iou_threshold:
                return True
        return False

    def create_ob(
        self, origin_idx: int, current_idx: int, is_high: bool,
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atrs: np.ndarray,
    ) -> Optional[Dict]:
        atr_val = atrs[origin_idx]
        if np.isnan(atr_val) or atr_val <= 0:
            return None
            
        atr_val *= self.atr_mult
        base = highs[origin_idx] if is_high else lows[origin_idx]
        if np.isnan(base):
            return None
            
        if is_high:
            top = base
            bot = base - atr_val
        else:
            top = base + atr_val
            bot = base
            
        if bot > top:
            top, bot = bot, top

        phantom_closes = closes[origin_idx + 1:current_idx]
        phantom_highs = highs[origin_idx + 1:current_idx]
        phantom_lows = lows[origin_idx + 1:current_idx]
        touches = 0
        
        if len(phantom_closes) > 0:
            if is_high:
                if (phantom_closes > top).any(): return None
            else:
                if (phantom_closes < bot).any(): return None
                
            intersections = (phantom_lows <= top) & (phantom_highs >= bot)
            touches = int(np.sum(intersections))
            
        return {'idx': origin_idx, 'top': top, 'bot': bot, 'touches': touches}

    def promote_or_create(
        self, origin_idx: int, current_idx: int, is_high: bool,
        internal_q: deque, swing_q: deque,
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atrs: np.ndarray,
    ) -> None:
        promoted_ob = None
        for ob in list(internal_q):
            if ob['idx'] == origin_idx:
                promoted_ob = ob
                break

        if promoted_ob:
            if not self.is_duplicate_spatial(promoted_ob, swing_q):
                internal_q.remove(promoted_ob)
                swing_q.append(promoted_ob)
        else:
            new_ob = self.create_ob(origin_idx, current_idx, is_high, highs, lows, closes, atrs)
            if new_ob and not self.is_duplicate_spatial(new_ob, swing_q):
                swing_q.append(new_ob)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        n = len(result)
        
        highs = np.ascontiguousarray(result['high'].to_numpy(), dtype=np.float64)
        lows = np.ascontiguousarray(result['low'].to_numpy(), dtype=np.float64)
        closes = np.ascontiguousarray(result['close'].to_numpy(), dtype=np.float64)
        atrs = np.ascontiguousarray(result['ATR'].to_numpy(), dtype=np.float64)

        int_ph = self.get_pivot_flags(result['high'], self.int_lb, is_high=True)
        int_pl = self.get_pivot_flags(result['low'], self.int_lb, is_high=False)
        swg_ph = self.get_pivot_flags(result['high'], self.swg_lb, is_high=True)
        swg_pl = self.get_pivot_flags(result['low'],  self.swg_lb, is_high=False)

        swg_bull = deque(maxlen=self.max_obs)
        swg_bear = deque(maxlen=self.max_obs)
        int_bull = deque(maxlen=self.max_obs)
        int_bear = deque(maxlen=self.max_obs)

        out_swg_supp = np.full(n, np.nan)
        out_swg_res = np.full(n, np.nan)
        out_int_supp = np.full(n, np.nan)
        out_int_res = np.full(n, np.nan)
        
        out_swg_supp_touches = np.zeros(n, dtype=np.float32)
        out_swg_res_touches = np.zeros(n, dtype=np.float32)
        
        mask_swg_supp = np.zeros(n, dtype=bool)
        mask_swg_res = np.zeros(n, dtype=bool)

        for i in range(self.max_window, n):
            curr_h = highs[i]
            curr_l = lows[i]
            curr_c = closes[i]
            
            if np.isnan(curr_h) or np.isnan(curr_l) or np.isnan(curr_c):
                continue
                
            if swg_ph[i]:
                self.promote_or_create(i - self.swg_lb, i, True, int_bear, swg_bear, highs, lows, closes, atrs)
            if swg_pl[i]:
                self.promote_or_create(i - self.swg_lb, i, False, int_bull, swg_bull, highs, lows, closes, atrs)

            if int_ph[i]:
                new_ob = self.create_ob(i - self.int_lb, i, True, highs, lows, closes, atrs)
                if new_ob and not self.is_duplicate_spatial(new_ob, swg_bear) and not self.is_duplicate_spatial(new_ob, int_bear):
                    int_bear.append(new_ob)
            if int_pl[i]:
                new_ob = self.create_ob(i - self.int_lb, i, False, highs, lows, closes, atrs)
                if new_ob and not self.is_duplicate_spatial(new_ob, swg_bull) and not self.is_duplicate_spatial(new_ob, int_bull):
                    int_bull.append(new_ob)

            for ob in swg_bull:
                if curr_l <= ob['top'] and curr_h >= ob['bot']: ob['touches'] += 1
            for ob in int_bull:
                if curr_l <= ob['top'] and curr_h >= ob['bot']: ob['touches'] += 1
            for ob in swg_bear:
                if curr_l <= ob['top'] and curr_h >= ob['bot']: ob['touches'] += 1
            for ob in int_bear:
                if curr_l <= ob['top'] and curr_h >= ob['bot']: ob['touches'] += 1

            for q, key, is_bull in [(swg_bull, 'bot', True), (int_bull, 'bot', True),
                                    (swg_bear, 'top', False), (int_bear, 'top', False)]:
                to_evict = [ob for ob in q if curr_c < ob[key]] if is_bull else [ob for ob in q if curr_c > ob[key]]
                for ob in to_evict:
                    q.remove(ob)

            if swg_bull:
                closest = max(swg_bull, key=lambda x: x['top'])
                out_swg_supp[i] = closest['top']
                out_swg_supp_touches[i] = closest['touches']
                mask_swg_supp[i] = True
            if swg_bear:
                closest = min(swg_bear, key=lambda x: x['bot'])
                out_swg_res[i] = closest['bot']
                out_swg_res_touches[i] = closest['touches']
                mask_swg_res[i] = True
            if int_bull:
                out_int_supp[i] = max(int_bull, key=lambda x: x['top'])['top']
            if int_bear:
                out_int_res[i] = min(int_bear, key=lambda x: x['bot'])['bot']

        result['SwingSupportTop'] = out_swg_supp
        result['SwingSupportTouches'] = out_swg_supp_touches
        result['ActiveSwgSupportMask'] = mask_swg_supp.astype(np.int8)

        result['SwingResistanceBot'] = out_swg_res
        result['SwingResistanceTouches'] = out_swg_res_touches
        result['ActiveSwgResistanceMask'] = mask_swg_res.astype(np.int8)

        dist_supp = (result['close'] - result['SwingSupportTop']) / result['close']
        dist_res = (result['SwingResistanceBot'] - result['close']) / result['close']
        
        result['DistSwingSuppPct'] = np.where(mask_swg_supp, dist_supp, self.missing_fill)
        result['DistSwingResPct'] = np.where(mask_swg_res, dist_res, self.missing_fill)
        
        return result

def calculate_stochastic(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14, smooth_k: int = 3) -> pd.Series:
    """Calculates Stochastic Oscillator %K, smoothed."""
    low_min = l.rolling(window=period, min_periods=period).min()
    high_max = h.rolling(window=period, min_periods=period).max()
    
    # Raw stochastic is 0 to 100
    stoch_k = 100 * ((c - low_min) / (high_max - low_min + EPS))
    return stoch_k.rolling(window=smooth_k, min_periods=smooth_k).mean()

def calculate_adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Average Directional Index using Wilder's Smoothing."""
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    
    pos_di = 100 * (pd.Series(pos_dm, index=h.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr)
    neg_di = 100 * (pd.Series(neg_dm, index=h.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr)
    
    dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + EPS))
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

def calculate_ichimoku_distances(h: pd.Series, l: pd.Series, c: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculates relative distance to Ichimoku Cloud components."""
    # Tenkan-sen (9) & Kijun-sen (26)
    tenkan = (h.rolling(9, min_periods=9).max() + l.rolling(9, min_periods=9).min()) / 2
    kijun = (h.rolling(26, min_periods=26).max() + l.rolling(26, min_periods=26).min()) / 2
    
    # Senkou Spans (Cloud) - Calculated historically
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (h.rolling(52, min_periods=52).max() + l.rolling(52, min_periods=52).min()) / 2
    
    # Shift forward by 26 so today's row uses the cloud projected 26 periods ago
    span_a = span_a_raw.shift(26)
    span_b = span_b_raw.shift(26)
    
    # Return as percentage distances from close (Unbounded, stationary)
    dist_tenkan = (c - tenkan) / c
    dist_kijun = (c - kijun) / c
    dist_span_a = (c - span_a) / c
    dist_span_b = (c - span_b) / c
    
    return dist_tenkan, dist_kijun, dist_span_a, dist_span_b

def calculate_features(
    df_raw: pd.DataFrame,
    momentum_period: int = 14,
    vol_short_period: int = 6,
    vol_long_period: int = 100,
    skew_period: int = 28,
    zscore_period: int = 50,
    icp_period: int = 14,
    mds_fast_window: int = 5,
    mds_slow_window: int = 30,
    vol_asym_window: int = 20,
    stoch_period: int = 14,
    adx_period: int = 14,
    ob_atr_mult: Optional[float] = None,
    add_session_features: bool = True,
    session: SessionConfig = SessionConfig(),
    clip_outside_session: bool = True,
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    cols = {c.lower() for c in df_raw.columns}
    missing = sorted(list(required - cols))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close']].astype(np.float64)

    c = df['close']
    h = df['high']
    l = df['low']
    log_ret = np.log(c / c.shift(1))

    out = pd.DataFrame(index=df.index)

    # --- 1. Existing Core Features ---

    # Efficiency Ratio (KER) - Naturally bounded [-1, 1]
    net_move = c.diff(momentum_period)
    path_len = c.diff().abs().rolling(momentum_period, min_periods=momentum_period).sum()
    ker_signed = (net_move.abs() / (path_len + EPS)) * np.sign(net_move)
    out["feat_efficiency"] = clip_scale(ker_signed, bound=1.0)

    # Volatility Regime - Unbounded Log Ratio
    v_short = log_ret.rolling(vol_short_period, min_periods=vol_short_period).std()
    v_long = log_ret.rolling(vol_long_period, min_periods=vol_long_period).std()
    out["feat_volatility_regime"] = np.log((v_short + EPS) / (v_long + EPS))

    # Intracandle Position (ICP) - Naturally bounded [-1, 1]
    raw_icp = (c - l) / (h - l + EPS)
    scaled_icp = (raw_icp * 2.0) - 1.0
    icp_smooth = scaled_icp.rolling(icp_period, min_periods=icp_period).mean()
    out["feat_icp"] = clip_scale(icp_smooth, bound=1.0)

    # RSI (Wilder's Smoothing) - Bounded [-1, 1]
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    alpha = 1.0 / float(momentum_period)
    avg_up = up.ewm(alpha=alpha, adjust=False, min_periods=momentum_period).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False, min_periods=momentum_period).mean()
    rs = avg_up / (avg_down + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_centered = (rsi - 50.0) / 50.0
    out["feat_momentum_rsi"] = clip_scale(rsi_centered, bound=1.0)

    # Skew - Unbounded
    out["feat_dist_skew"] = log_ret.rolling(skew_period, min_periods=skew_period).skew()

    # Z-Score - Unbounded
    ret_mean = log_ret.rolling(zscore_period, min_periods=zscore_period).mean()
    ret_std = log_ret.rolling(zscore_period, min_periods=zscore_period).std()
    out["feat_zscore"] = (log_ret - ret_mean) / (ret_std + EPS)

    # Momentum Divergence Score - Unbounded
    out["feat_momentum_mds"] = momentum_divergence_score(
        log_ret=log_ret, fast_window=mds_fast_window, slow_window=mds_slow_window
    )

    # Volatility Asymmetry - Unbounded
    out["feat_vol_asymmetry"] = directional_vol_asymmetry(
        log_ret=log_ret, window=vol_asym_window
    )

    # --- 2. Build ATR for Order Blocks ---
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean()

    # --- 3. RUN ORDER BLOCK ENGINE ---
    # Use config default if none provided to function
    atr_m = ob_atr_mult if ob_atr_mult is not None else OB_ATR_MULT
    ob_engine = OptimizedOrderBlockEngine(
        internal_lookback=5, swing_lookback=20, atr_multiplier=atr_m, missing_value_fill=5.0
    )
    ob_df = ob_engine.generate_features(df)

    # --- 4. Order Block Features ---
    # Unbounded distance percentages
    out["feat_ob_dist_supp"] = ob_df['DistSwingSuppPct']
    out["feat_ob_dist_res"] = ob_df['DistSwingResPct']

    # Bounded touches (Smooth Monotonic Unidirectional Scaling)
    # Replaces hard clipping. 1 touch = ~0.32, 3 touches = ~0.76, 10 touches = ~0.99
    out["feat_ob_supp_touches"] = np.tanh(ob_df['SwingSupportTouches'] / 3.0)
    out["feat_ob_res_touches"] = np.tanh(ob_df['SwingResistanceTouches'] / 3.0)

    # Binary active masks (0/1)
    out["feat_ob_supp_active"] = ob_df['ActiveSwgSupportMask'].astype(np.float32)
    out["feat_ob_res_active"] = ob_df['ActiveSwgResistanceMask'].astype(np.float32)

    # -------------------------------------------------------------------------
    # --- NEW INDICATORS: Stochastic, ADX, Ichimoku ---
    # -------------------------------------------------------------------------

    # 1. Stochastic (Centered & Bounded)
    raw_stoch = calculate_stochastic(h, l, c, period=stoch_period)
    stoch_centered = (raw_stoch - 50.0) / 50.0  # Centers at 0, maps 0-100 to -1 to 1
    out['feat_momentum_stoch'] = clip_scale(stoch_centered, bound=1.0)
    # TITLE: Stochastic Oscillator - Bounded [-1, 1]

    # 2. ADX (Empirical Centering & Monotonic Compression)
    raw_adx = calculate_adx(h, l, c, period=adx_period)

    # Center at 25 (empirical trend threshold) instead of 50.
    # A value of 25 = 0.0 (Neutral to GP). 
    # Wrapped in Tanh to smoothly bound mega-trends (>50) inside [0, 1)
    adx_centered = (raw_adx - 25.0) / 25.0 
    out['feat_trend_adx'] = np.tanh(adx_centered)
    # TITLE: ADX Trend Strength - Bounded [-1, 1]

    # 3. Ichimoku Distances (Unbounded percentage distances)
    d_tenkan, d_kijun, d_span_a, d_span_b = calculate_ichimoku_distances(h, l, c)
    out['feat_ichimoku_dist_tenkan'] = d_tenkan
    out['feat_ichimoku_dist_kijun'] = d_kijun
    out['feat_ichimoku_dist_span_a'] = d_span_a
    out['feat_ichimoku_dist_span_b'] = d_span_b
    # TITLE: Ichimoku Cloud Distances - Unbounded

    # --- 5. Session Features ---
    if add_session_features:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError("add_session_features=True requires a DatetimeIndex")
        s_sin, s_cos = session_cyclic_position(
            index=out.index, session=session, clip_outside_session=clip_outside_session,
        )
        out["feat_session_sin"] = s_sin.astype(np.float32)
        out["feat_session_cos"] = s_cos.astype(np.float32)

    # Final cleanup
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.astype(dtype, copy=False)
    
    del df
    del ob_df
    gc.collect()

    return out
