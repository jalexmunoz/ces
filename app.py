import os, math
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ces_scoring_v1 import run_ces  # tu módulo ya probado

app = FastAPI(title="CES API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- util: convertir a tipos nativos JSON ----------
def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.generic):           # numpy scalar
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/csv")
def debug_csv(path: Optional[str] = None):
    csv_path = path or os.getenv("INPUT_CSV", "ces_indicators_today_example.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")
    last = df.dropna(how="all").iloc[0].to_dict()
    return JSONResponse(content=to_py(last))

@app.get("/ces")
def ces_from_csv(path: Optional[str] = None):
    """
    Lee el CSV (1 fila de indicadores) y devuelve el CES Score.
    Usa ?path=... o la env var INPUT_CSV. Si no, ces_indicators_today_example.csv.
    """
    try:
        csv_path = path or os.getenv("INPUT_CSV", "ces_indicators_today_example.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
        indicators = df.iloc[0].to_dict()
        result = run_ces(indicators)
        return JSONResponse(content=to_py(result))
    except HTTPException:
        raise
    except Exception as e:
        # imprime detalle en consola y devuelve 500 claro
        print("[/ces] error:", repr(e))
        raise HTTPException(status_code=500, detail=f"CES error: {e}")

@app.post("/ces")
async def ces_from_json(payload: Dict[str, Any]):
    if "indicators" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'indicators'")
    try:
        result = run_ces(payload["indicators"], weights=payload.get("weights"))
        return JSONResponse(content=to_py(result))
    except Exception as e:
        print("[/ces POST] error:", repr(e))
        raise HTTPException(status_code=500, detail=f"CES error: {e}")

@app.post("/ces/upload")
async def ces_from_uploaded_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
        indicators = df.iloc[0].to_dict()
        result = run_ces(indicators)
        return JSONResponse(content=to_py(result))
    except HTTPException:
        raise
    except Exception as e:
        print("[/ces/upload] error:", repr(e))
        raise HTTPException(status_code=400, detail=f"Bad CSV: {e}")

