# ces_scoring_v1.py
import json, math
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

DEFAULT_WEIGHTS = {
    "etf_netflows_5dma": 15, "oi_mcap": 12, "basis_annualized": 12,
    "funding_8h_3d": 8, "cvd_90d": 10, "btc_dominance_delta_7d": 8,
    "fear_greed": 8, "weekly_rsi": 7, "mayer_multiple": 7,
    "pi_cycle_top": 5, "cbbi": 5, "global_m2": 13,
}

DEFAULT_SCALES = {
    "fear_greed": {"type":"direct","dir":"pos"},
    "cbbi": {"type":"direct","dir":"pos"},
    "weekly_rsi": {"type":"direct","dir":"pos"},
    "pi_cycle_top": {"type":"linear","min":0.8,"max":2.4,"dir":"pos"},
    "mayer_multiple": {"type":"linear","min":0.7,"max":2.4,"dir":"pos"},
    "basis_annualized": {"type":"linear","min":0.0,"max":20.0,"dir":"pos"},
    "funding_8h_3d": {"type":"linear","min":-0.02,"max":0.10,"dir":"pos"},
    "oi_mcap": {"type":"linear","min":0.008,"max":0.025,"dir":"pos"},
    "etf_netflows_5dma": {"type":"linear","min":-2.0,"max":5.0,"dir":"pos"},
    "btc_dominance_delta_7d": {"type":"linear","min":-3.0,"max":3.0,"dir":"neg"},
    "global_m2": {"type":"linear","min":-5.0,"max":10.0,"dir":"pos"},
    "cvd_90d": {"type":"linear","min":-3.0,"max":3.0,"dir":"pos"},
}

ACTION_RULES = [
    (70, 2, "SELL HARD", "25–40%", "Score alto y sobre-extensión"),
    (55, 0, "SELL PARTIAL", "10–20%", "Riesgo elevado"),
    (45, 0, "TIGHTEN", "Stops/rotar a BTC/ETH", "Riesgo medio"),
    (0,  0, "HOLD", "Sin cambios", "Riesgo contenido"),
]
FROTH_SIGNALS = ["basis_annualized","funding_8h_3d","oi_mcap"]

def _nan(x): 
    return x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x)))
def _clamp01(x: float) -> float:
    return 0.0 if np.isnan(x) else max(0.0, min(1.0, x))

def _linear(x, vmin, vmax, direction):
    if _nan(x): return np.nan
    t = (x - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    t = _clamp01(t)
    if direction == "neg": t = 1.0 - t
    return t*100.0

def _direct(x, direction):
    if _nan(x): return np.nan
    t = _clamp01(x/100.0)
    if direction == "neg": t = 1.0 - t
    return t*100.0

def _normalize(name, value, scales):
    rule = scales.get(name)
    if not rule: return np.nan
    if rule["type"]=="direct": return _direct(value, rule.get("dir","pos"))
    if rule["type"]=="linear": return _linear(value, rule["min"], rule["max"], rule.get("dir","pos"))
    return np.nan

def _count_froth(readings):
    return sum(1 for k in FROTH_SIGNALS if not np.isnan(readings.get(k,np.nan)) and readings[k] >= 70)

def _composite(readings, weights):
    use = {k:v for k,v in readings.items() if not np.isnan(v) and k in weights}
    if not use: return 0.0, {}
    total_w = sum(weights[k] for k in use)
    score = 0.0; contrib={}
    for k,v in use.items():
        w = (weights[k]/total_w)*100.0
        contrib[k] = v*(w/100.0)
        score += contrib[k]
    return score, contrib

def _decide(score, readings):
    extra = _count_froth(readings)
    for thr,need,act,size,note in ACTION_RULES:
        if score>=thr and extra>=need:
            return {"action":act,"size":size,"note":note,"froth_signals":extra}
    return {"action":"HOLD","size":"Sin cambios","note":"—","froth_signals":extra}

def run_ces(indicators: Dict[str,Any],
            weights: Optional[Dict[str,int]]=None,
            scales: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    weights = weights or DEFAULT_WEIGHTS
    scales  = scales  or DEFAULT_SCALES
    readings_scaled = {k:_normalize(k, indicators.get(k,np.nan), scales) for k in weights}
    score, _ = _composite(readings_scaled, weights)
    act = _decide(score, readings_scaled)
    return {
        "readings_raw": indicators,
        "readings_scaled": readings_scaled,
        "weights_used": {k:weights[k] for k in readings_scaled if not np.isnan(readings_scaled[k])},
        "score": round(float(score),2),
        "action": act["action"],
        "action_meta": {"size":act["size"],"froth_signals":act["froth_signals"],"note":act["note"]},
    }
