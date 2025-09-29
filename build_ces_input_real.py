# build_ces_input_real.py V 2.0
import os
import argparse
import warnings
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
import math
# Adapters
from adapters.derivs import (
    binance_oi_hist,
    dapi_continuous_quarterly_klines,
    um_continuous_quarterly_klines,
)
from adapters.sosovalue import get_ma5_for_market
# Dominancia diaria pre-actualizada por scripts/update_dom_history.py
# (guardamos en data/dom_history.csv)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOM_CSV = os.path.join(DATA_DIR, "dom_history.csv")

warnings.filterwarnings("ignore")

def _utc_date(s: str | None):
    return pd.to_datetime(s, utc=True) if s else None

def _ensure_daily_index(df: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    lo = start.normalize() if start is not None else df.index.min().normalize()
    hi = end.normalize() if end is not None else df.index.max().normalize()
    return pd.date_range(lo, hi, freq="D", tz="UTC")

def _safe_tz_localize(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Evita el "Already tz-aware"
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")

def compute_basis_ann(start, end) -> pd.Series:
    """
    basis_ann = ((FUT / SPOT) - 1) * (365/90) * 100
    FUT: CURRENT_QUARTER, prefer Coin-M (BTCUSD), fallback USDT-M (BTCUSDT)
    SPOT proxy: USDT-M PERPETUAL (BTCUSDT)
    """
    fut = dapi_continuous_quarterly_klines("BTCUSD", "1d", start, end, "CURRENT_QUARTER")
    if fut is None or fut.empty:
        fut = um_continuous_quarterly_klines("BTCUSDT", "1d", start, end, "CURRENT_QUARTER")

    spot = um_continuous_quarterly_klines("BTCUSDT", "1d", start, end, "PERPETUAL")
    if fut is None or fut.empty or spot is None or spot.empty:
        return pd.Series(dtype=float, name="basis_ann")

    fut = fut[["close"]].rename(columns={"close": "fut_close"})
    spot = spot[["close"]].rename(columns={"close": "spot_close"})
    df = fut.join(spot, how="inner")

    # UTC-safe index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[~df.index.duplicated()].sort_index()

    ann_factor = 365.0 / 90.0
    basis = (df["fut_close"] / df["spot_close"] - 1.0) * ann_factor * 100.0
    basis.name = "basis_ann"
    return basis

def compute_oi_mcap_ratio(start, end) -> pd.Series:
    # OI (USDT-M) daily, deep backfill
    oi = binance_oi_hist("BTCUSDT", "1d", start, end)
    if oi.empty:
        return pd.Series(dtype=float, name="oi_mcap_ratio")

    # Spot proxy from perp (USDT-M PERPETUAL) for market-cap proxy
    price = um_continuous_quarterly_klines("BTCUSDT", "1d", start, end, "PERPETUAL")[["close"]]
    if price.empty:
        return pd.Series(dtype=float, name="oi_mcap_ratio")

    # Normalize both to daily UTC midnight to avoid “almost-daily” index mismatches
    oi.index = oi.index.tz_convert("UTC").normalize()
    price.index = price.index.tz_convert("UTC").normalize()
    oi_d = oi.groupby(oi.index).last()
    px_d = price.groupby(price.index).last()

    SUPPLY_BTC = 19_700_000  # proxy (safe enough for the ratio)
    mcap = (px_d["close"] * SUPPLY_BTC).rename("mcap")
    df = oi_d.join(mcap, how="inner")
    return (df["oi"] / df["mcap"]).rename("oi_mcap_ratio")

def compute_breadth_topN_sma20(symbols: list[str], start, end) -> pd.Series:
    """
    Marca 1 si el cierre > SMA20; luego porcentaje de top-N en 1.
    Para empezar, computo sobre una lista fija de ~Top 37 que ya estás recorriendo.
    """
    def _kl(pair):
        return um_continuous_quarterly_klines(pair, "1d", start, end, "PERPETUAL")[["close"]]

    signals = []
    for i, s in enumerate(symbols, 1):
        print(f"  ({i}/{len(symbols)}) {s}")
        kl = _kl(s)
        if kl.empty:
            continue
        ma = kl["close"].rolling(20, min_periods=5).mean()
        sig = (kl["close"] > ma).astype(int).rename(s)
        signals.append(sig)

    if not signals:
        return pd.Series(dtype=float, name="breadth_topN_sma20")

    panel = pd.concat(signals, axis=1)
    breadth = panel.mean(axis=1, skipna=True).rename("breadth_topN_sma20")
    return breadth

def load_dom_history() -> pd.Series:
    fname = os.getenv("DOM_CSV", "cmc_dom_history.csv")
    candidates = [fname, os.path.join(os.path.dirname(__file__), fname)]
    for p in candidates:
        if os.path.exists(p):
            dom = pd.read_csv(p)
            # Robust parse
            if "date" not in dom.columns:
                # try first column as date
                dom.columns = ["date"] + dom.columns.tolist()[1:]
            dom["date"] = pd.to_datetime(dom["date"], utc=True)
            dom = dom.set_index("date").sort_index()
            # column could be e.g. btc_dominance_pct / dominance / btc_dominance
            col = next((c for c in dom.columns if "domin" in c.lower()), None)
            if not col:
                return pd.Series(dtype=float, name="btc_dom_7d_delta")
            s = pd.to_numeric(dom[col], errors="coerce")
            return (s - s.shift(7)).rename("btc_dom_7d_delta")
    return pd.Series(dtype=float, name="btc_dom_7d_delta")

def compute_cvd_proxy_90d(symbols_1h: list[str], start, end) -> pd.Series:
    """
    CVD proxy simple a 1H: suma de (close - open) * volume como proxy de balance buy-sell,
    acumulado y luego variación 90D. (Mantengo simple y robusto.)
    """
    from adapters.derivs import um_continuous_quarterly_klines as kl

    def _cvd_for(pair: str) -> pd.Series:
        h = kl(pair, "1h", start, end, "PERPETUAL")
        if h.empty:
            return pd.Series(dtype=float)
        h.index = _safe_tz_localize(h.index)
        if "open" in h.columns:
            delta = (h["close"] - h["open"]).fillna(0.0)
        else:
            # Fallback robusto si no hay 'open'
            delta = h["close"].diff().fillna(0.0)
        cvd = delta.cumsum()
        return cvd

    parts = []
    for s in symbols_1h:
        try:
            parts.append(_cvd_for(s))
        except Exception as e:
            print(f"  Error en CVD para {s}: {e}")

    if not parts:
        return pd.Series(dtype=float, name="cvd_90d")

    agg = pd.concat(parts, axis=1).mean(axis=1)
    cvd_90d = (agg - agg.shift(24*90)).rename("cvd_90d")  # 90 días * 24h
    return cvd_90d

def compute_funding_8h_3d(start, end) -> pd.Series:
    """
    Placeholder: si ya tienes el funding real, conecta aquí.
    Por ahora, devuelvo una serie vacía para no romper el pipeline.
    """
    return pd.Series(dtype=float, name="funding_8h_3d")
def to_utc(ts_or_str, default=None):
    """
    Convierte start/end a pandas.Timestamp en UTC (tz-aware).
    Acepta: None, str, datetime, pandas.Timestamp.
    """
    if ts_or_str is None:
        return default
    ts = pd.Timestamp(ts_or_str)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def main(start_str: str | None, end_str: str | None, top_n: int, out_csv: str, enable_cvd: bool):
    print("--- CES 3.0 Builder (v13) ---")

    load_dotenv()  # asegúrate que .env fue leído

    # --- MANEJO DE FECHAS ROBUSTO Y CENTRALIZADO ---
    # 1. Se convierten las fechas de entrada a Timestamps UTC-aware.
    now_utc = pd.Timestamp.now(tz="UTC")
    start_ts = to_utc(start_str) or pd.Timestamp("2022-01-01", tz="UTC")
    end_ts = to_utc(end_str, default=now_utc)

    # 2. Se crean las versiones en string para las funciones que las necesiten.
    start_iso = start_ts.strftime("%Y-%m-%d")
    end_iso = end_ts.strftime("%Y-%m-%d")
    
    # ---------- OI / Mcap ----------
    print("Descargando OI/Mcap...")
    # Se usan las variables Timestamp (start_ts, end_ts)
    oi_mcap = compute_oi_mcap_ratio(start_ts, end_ts)

    # ---------- Basis ----------
    print("Calculando Basis (DAPI → UMFutures fallback)...")
    basis = compute_basis_ann(start_ts, end_ts)

    # ---------- Breadth ----------
    print("Calculando Breadth (Top-N sobre SMA20)...")
    top_symbols = [
        "BTCUSDT","ETHUSDT","XRPUSDT","BNBUSDT","SOLUSDT","TRXUSDT","DOGEUSDT","ADAUSDT","LINKUSDT",
        "SUIUSDT","XLMUSDT","BCHUSDT","AVAXUSDT","HBARUSDT","LTCUSDT","TONUSDT","SHIBUSDT","DOTUSDT",
        "UNIUSDT","AAVEUSDT","ENAUSDT","PEPEUSDT","ETCUSDT","TAOUSDT","NEARUSDT","APTUSDT","ONDOUSDT",
        "ICPUSDT","ARBUSDT","POLUSDT","ATOMUSDT","VETUSDT","ALGOUSDT","WLDUSDT","RENDERUSDT"
    ][:max(1, top_n)]
    breadth = compute_breadth_topN_sma20(top_symbols, start_ts, end_ts)

    # ---------- Dominancia (7d delta) ----------
    print("Dominancia y Macro...")
    dom_delta = load_dom_history()

    # ---------- ETF Flows (SoSoValue) ----------
    print("Flujos ETF (SoSoValue)...")
    # Ya no hay código de fechas aquí, usamos directamente start_iso y end_iso.
    try:
        btc_ma5 = get_ma5_for_market("us-btc-spot", start_iso, end_iso).rename(columns={"ma5": "etf_btc_ma5"})
    except Exception as e:
        print(f"ADVERTENCIA: fallo en SoSoValue BTC: {e}")
        btc_ma5 = pd.DataFrame(columns=["etf_btc_ma5"])

    try:
        eth_ma5 = get_ma5_for_market("us-eth-spot", start_iso, end_iso).rename(columns={"ma5": "etf_eth_ma5"})
    except Exception as e:
        print(f"ADVERTENCIA: fallo en SoSoValue ETH: {e}")
        eth_ma5 = pd.DataFrame(columns=["etf_eth_ma5"])

    # ---------- CVD (opcional) ----------
    if enable_cvd:
        print("Calculando CVD proxy (puede tardar)...")
        cvd = compute_cvd_proxy_90d(["BTCUSDT","ETHUSDT","BNBUSDT"], start_ts, end_ts)
    else:
        cvd = pd.Series(dtype=float, name="cvd_90d")

    # ---------- Ensamble y saneo final ----------
    # índice maestro
    pieces = [oi_mcap, basis, breadth, dom_delta, cvd]
    for x in [btc_ma5["etf_btc_ma5"] if not btc_ma5.empty else None,
              eth_ma5["etf_eth_ma5"] if not eth_ma5.empty else None]:
        if x is not None:
            pieces.append(x)

    frame = pd.concat(pieces, axis=1)
    min_date = start_ts or frame.index.min()
    max_date = end_ts or frame.index.max()
    idx = _ensure_daily_index(frame, min_date, max_date)
    frame = frame.reindex(idx)
    
    # Nota: Si _ensure_daily_index no maneja bien los Timestamps, podrías necesitar
    # pasarle las fechas como string: _ensure_daily_index(frame, min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
    
    # Limpieza de tipos (evita FutureWarning)
    for c in frame.columns:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    # Telemetría de cobertura
    print("\n--- [TELEMETRÍA DE COBERTURA] ---")
    for col in ["funding_8h_3d","oi_mcap_ratio","basis_ann","breadth_topN_sma20","etf_btc_ma5","etf_eth_ma5","btc_dom_7d_delta","cvd_90d"]:
        if col in frame:
            cov = frame[col].notna().mean() * 100
            print(f"{col:>20}: {cov:5.1f}%")
        else:
            print(f"{col:>20}:    0.0%")

    # Persistencia
    frame.to_csv(out_csv, index_label="date")
    print(f"\nWrote {out_csv} with shape {frame.shape}")
    
    # ===== CES: exportar una sola fila para el panel/API =====
    
        # === CES one-row export (self-contained, no external deps) ===
        
    # >>> CES export utils (global)
import os, math
import pandas as pd

def _ces_safe(x):
    try:
        if x is None:
            return ""
        if isinstance(x, (float, int)) and (math.isnan(x) or math.isinf(x)):
            return ""
        return float(x)
    except Exception:
        return ""

def export_ces_indicators_csv(path: str, **kwargs) -> str:
    """
    Crea un CSV de una fila con los indicadores CES.
    """
    cols = [
        "fear_greed","cbbi","weekly_rsi","pi_cycle_top","mayer_multiple",
        "basis_annualized","funding_8h_3d","oi_mcap","etf_netflows_5dma",
        "btc_dominance_delta_7d","global_m2","cvd_90d",
    ]
    row = {c: _ces_safe(kwargs.get(c, "")) for c in cols}
    pd.DataFrame([row], columns=cols).to_csv(path, index=False)
    return path
# <<< CES export utils

    ces_export = os.getenv("CES_EXPORT", r"C:\Criptos\CES\ces_indicators_today.csv")
    try:
        pd.DataFrame([{k: _ces_safe(v) for k, v in row.items()}], columns=cols).to_csv(ces_export, index=False)
        print(f"[CES] CSV actualizado → {ces_export}")
    except Exception as e:
        print(f"[CES] No se pudo exportar el CSV CES: {e}")
    # === /CES ===


try:
    export_ces_indicators_csv(
        ces_export,
        fear_greed=fear_greed, cbbi=cbbi, weekly_rsi=weekly_rsi,
        pi_cycle_top=pi_cycle_top, mayer_multiple=mayer_multiple,
        basis_annualized=basis_annualized, funding_8h_3d=funding_8h_3d,
        oi_mcap=oi_mcap, etf_netflows_5dma=etf_netflows_5dma,
        btc_dominance_delta_7d=btc_dominance_delta_7d, global_m2=global_m2,
        cvd_90d=cvd_90d,
    )
    print(f"[CES] CSV actualizado → {ces_export}")
except Exception as e:
    print(f"[CES] No se pudo exportar el CSV CES: {e}")
# ===== /CES =====

    
    # ===== CES: exportar una sola fila para el panel/API =====
def _last_valid(obj):
    try:
        s = obj.dropna()
        return float(s.iloc[-1]) if not s.empty else ""
    except Exception:
        return ""

# Toma lo que ya calcula tu builder (si no existe, deja en blanco)
fear_greed = ""                  # conéctalo cuando lo tengas
cbbi = ""                        # idem
weekly_rsi = ""                  # idem
pi_cycle_top = ""                # idem
mayer_multiple = ""              # idem

basis_annualized = _last_valid(basis) if "basis" in locals() else ""
funding_8h_3d = ""               # conéctalo cuando lo tengas
oi_mcap = _last_valid(oi_mcap) if "oi_mcap" in locals() else ""

etf_netflows_5dma = ""
if "frame" in locals():
    if "etf_btc_ma5" in frame.columns:              # SoSoValue MA5
        etf_netflows_5dma = _last_valid(frame["etf_btc_ma5"])
btc_dominance_delta_7d = ""
if "frame" in locals():
    if "btc_dom_7d_delta" in frame.columns:
        btc_dominance_delta_7d = _last_valid(frame["btc_dom_7d_delta"])

global_m2 = ""                 # conéctalo cuando lo tengas
cvd_90d = _last_valid(cvd) if "cvd" in locals() else ""

# Dónde se guarda (puedes cambiar con la env var CES_EXPORT)
ces_export = os.getenv("CES_EXPORT", r"C:\Criptos\CES\ces_indicators_today.csv")

try:
    export_ces_indicators_csv(
        ces_export,
        fear_greed=fear_greed, cbbi=cbbi, weekly_rsi=weekly_rsi,
        pi_cycle_top=pi_cycle_top, mayer_multiple=mayer_multiple,
        basis_annualized=basis_annualized, funding_8h_3d=funding_8h_3d,
        oi_mcap=oi_mcap, etf_netflows_5dma=etf_netflows_5dma,
        btc_dominance_delta_7d=btc_dominance_delta_7d, global_m2=global_m2,
        cvd_90d=cvd_90d,
    )
    print(f"[CES] CSV actualizado → {ces_export}")
except Exception as e:
    print(f"[CES] No se pudo exportar el CSV CES: {e}")
# ===== /CES =====


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=37)
    parser.add_argument("--out", type=str, default="ces_input_real.csv")
    parser.add_argument("--enable_cvd", action="store_true", default=False)
    args = parser.parse_args()
    main(args.start, args.end, args.top_n, args.out, args.enable_cvd)
def _safe(x):
    try:
        if x is None:
            return ""
        if isinstance(x, (float, int)) and (math.isnan(x) or math.isinf(x)):
            return ""
        return float(x)
    except Exception:
        return ""

def export_ces_indicators_csv(path: str, **kwargs) -> str:
    """
    Crea un CSV de una fila con los indicadores CES.
    """
    cols = [
        "fear_greed","cbbi","weekly_rsi","pi_cycle_top","mayer_multiple",
        "basis_annualized","funding_8h_3d","oi_mcap","etf_netflows_5dma",
        "btc_dominance_delta_7d","global_m2","cvd_90d",
    ]
    row = {c: _safe(kwargs.get(c, "")) for c in cols}
    pd.DataFrame([row], columns=cols).to_csv(path, index=False)
    return path
# ===== /CES export utils =====
