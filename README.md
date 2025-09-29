# CES – Composite Exposure Signal

Pipeline completo:
- **Builder** → `build_ces_input_real.py` (genera `ces_input_real.csv`).
- **Exporter** → `ces_export_one_row.py` (toma el último válido por columna y escribe `ces_indicators_today.csv`).
- **API** → `app.py` (FastAPI `/ces`).
- **Panel** → `ces_panel.py` (Streamlit).

## Requisitos
```bash
pip install -r requirements.txt
