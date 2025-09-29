import argparse
import pandas as pd
import math
import os

OUT_COLS = [
    "fear_greed","cbbi","weekly_rsi","pi_cycle_top","mayer_multiple",
    "basis_annualized","funding_8h_3d","oi_mcap","etf_netflows_5dma",
    "btc_dominance_delta_7d","global_m2","cvd_90d",
]

# Mapeo: columna en tu CSV grande -> campo del CES de 1 fila
SOURCE_MAP = {
    "basis_ann":              "basis_annualized",
    "oi_mcap_ratio":          "oi_mcap",
    "etf_btc_ma5":            "etf_netflows_5dma",
    "btc_dom_7d_delta":       "btc_dominance_delta_7d",
    "cvd_90d":                "cvd_90d",
    # Si más adelante agregas columnas reales para estos, añádelas aquí:
    # "fear_greed_src":       "fear_greed",
    # "cbbi_src":             "cbbi",
    # "weekly_rsi_src":       "weekly_rsi",
    # "pi_cycle_top_src":     "pi_cycle_top",
    # "mayer_multiple_src":   "mayer_multiple",
    # "funding_8h_3d_src":    "funding_8h_3d",
    # "global_m2_src":        "global_m2",
}

def _safe(x):
    try:
        if x is None: return ""
        if isinstance(x, (float, int)) and (math.isnan(x) or math.isinf(x)): return ""
        return float(x)
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV grande (salida de build_ces_input_real.py)")
    ap.add_argument("--out", default=r"C:\Criptos\CES\ces_indicators_today.csv", help="CSV de 1 fila para el panel/API")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit("El CSV de entrada está vacío.")

    # Tomar la última fila no vacía
    last = df.dropna(how="all").iloc[-1].to_dict()

    # Construir fila CES
    row = {k: "" for k in OUT_COLS}
    for src_col, out_col in SOURCE_MAP.items():
        if src_col in last and last[src_col] == last[src_col]:
            row[out_col] = _safe(last[src_col])

    # Escribir
    pd.DataFrame([row], columns=OUT_COLS).to_csv(args.out, index=False)
    print(f"[CES] CSV de 1 fila escrito en: {args.out}")

if __name__ == "__main__":
    main()
