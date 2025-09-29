import requests
import streamlit as st
from datetime import datetime

API_URL = st.secrets.get("CES_API_URL", "http://localhost:8000/ces")

st.set_page_config(page_title="CES Dashboard", layout="wide")
st.title("🧭 CES Dashboard")

@st.cache_data(ttl=60)
def fetch_ces(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

col = st.sidebar
col.write("Settings")
api_url = col.text_input("CES endpoint", API_URL)
auto = col.toggle("Auto-refresh (60s cache)", value=True)

try:
    data = fetch_ces(api_url)
    score = data.get("score", 0)
    action = data.get("action", "N/A")
    meta = data.get("action_meta", {})
    size = meta.get("size", "—")
    froth = meta.get("froth_signals", 0)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Composite Score", f"{score:.2f}")
    c2.metric("Action", action)
    c3.metric("Size", size)
    c4.metric("Froth signals ≥70", froth)

    st.subheader("Readings (scaled 0–100)")
    scaled = data.get("readings_scaled", {})
    rows = sorted(scaled.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    st.dataframe(
        [{"indicator": k, "scaled": v} for k, v in rows],
        use_container_width=True, height=360
    )

    with st.expander("Raw readings & weights", expanded=False):
        st.write({"readings_raw": data.get("readings_raw", {})})
        st.write({"weights_used": data.get("weights_used", {})})

    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    st.error(f"Could not fetch CES data: {e}")
    st.info("Verify the server is running:\n"
            "1) In another terminal: uvicorn app:app --host 0.0.0.0 --port 8000 --reload\n"
            "2) Test: curl http://localhost:8000/health")

