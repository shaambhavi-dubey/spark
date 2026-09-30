"""
Phase 6: Live dashboard.

Reads directly from Redis (the same keys consumer/score_transactions.py
writes to) and displays a live-updating view: recent transactions, running
counters, and latency. Doesn't touch Kafka or the model at all -- purely a
window onto data the consumer is already producing.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import redis
import json
import pandas as pd
import time

import os

RECENT_TXN_KEY = "dashboard:recent_transactions"
STATS_KEY = "dashboard:stats"

st.set_page_config(page_title="UPI Fraud Detection - Live", layout="wide")

# Support Streamlit secrets (for Streamlit Cloud), env vars (for Docker), with fallback to localhost
def get_config(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

redis_host = get_config("REDIS_HOST", "localhost")
redis_port = int(get_config("REDIS_PORT", 6379))
redis_password = get_config("REDIS_PASSWORD", None)
redis_ssl = str(get_config("REDIS_SSL", "false")).lower() == "true"

r = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    ssl=redis_ssl,
    decode_responses=True
)

st.title("UPI Fraud Detection -- Live Pipeline")
st.caption("Kafka -> Redis (GraphSAGE embeddings) -> XGBoost -- synthetic transaction stream")

placeholder = st.empty()

while True:
    with placeholder.container():
        try:
            stats = r.hgetall(STATS_KEY)
            total_processed = int(stats.get("total_processed", 0))
            total_flagged = int(stats.get("total_flagged", 0))
            total_unknown = int(stats.get("total_unknown", 0))
            last_latency = float(stats.get("last_latency_ms", 0))
        except Exception as e:
            st.warning(f"Connecting to Redis at {redis_host}:{redis_port}... ({e})")
            time.sleep(2)
            continue

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Processed", total_processed)
        col2.metric("Flagged as Fraud", total_flagged)
        col3.metric("Unknown Accounts", total_unknown)
        col4.metric("Last Latency (ms)", f"{last_latency:.2f}")

        raw_entries = r.lrange(RECENT_TXN_KEY, 0, 49)
        entries = [json.loads(e) for e in raw_entries]

        if entries:
            df = pd.DataFrame(entries)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df[["transaction_id", "from_account", "to_account", "amount",
                     "score", "flag", "latency_ms", "timestamp"]]

            def highlight_flagged(row):
                if row["flag"] == "FLAGGED":
                    return ["background-color: #ffcccc"] * len(row)
                elif row["flag"] == "UNKNOWN ACCOUNT":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            st.subheader("Recent Transactions")
            st.dataframe(df.style.apply(highlight_flagged, axis=1), use_container_width=True, height=600)

            st.subheader("Latency (last 50 transactions)")
            st.line_chart(df.set_index("transaction_id")["latency_ms"])
        else:
            st.info("Waiting for transactions... make sure producer and consumer are running.")

    time.sleep(1)