"""
Phase 4 (with dashboard feed): Kafka consumer + live scorer.

Same as before, but now also pushes each result into Redis so the dashboard
(dashboard/app.py) can display live data without touching Kafka directly.
"""

import json
import time
import numpy as np
import redis
from kafka import KafkaConsumer
from xgboost import XGBClassifier

import os

RECENT_TXN_KEY = "dashboard:recent_transactions"
RECENT_TXN_MAX = 50
STATS_KEY = "dashboard:stats"


def load_model(path=None):
    if path is None:
        path = os.getenv("MODEL_PATH", "xgb_fraud_model.json")
        if not os.path.exists(path):
            path = "embeddings/xgb_fraud_model.json"
    model = XGBClassifier()
    model.load_model(path)
    return model


def score_transaction(r, model, transaction):
    from_id = transaction["from_account"]
    to_id = transaction["to_account"]
    amount = transaction["amount"]

    from_embedding_raw = r.get(f"embedding:{from_id}")
    to_embedding_raw = r.get(f"embedding:{to_id}")

    if from_embedding_raw is None or to_embedding_raw is None:
        return {"score": None, "reason": "unknown_account", "known": False}

    from_embedding = np.array(json.loads(from_embedding_raw))
    to_embedding = np.array(json.loads(to_embedding_raw))

    features = np.concatenate([from_embedding, to_embedding, [amount]]).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]

    return {"score": float(probability), "reason": "scored", "known": True}


def push_to_dashboard(r, transaction, result, elapsed_ms, flag):
    """
    Writes this transaction's result into a Redis list the dashboard reads
    from, and updates running counters. Kept separate from scoring logic so
    the dashboard is purely a consumer of this data, never a bottleneck for
    the actual scoring path.
    """
    entry = {
        "transaction_id": transaction["transaction_id"],
        "from_account": transaction["from_account"],
        "to_account": transaction["to_account"],
        "amount": transaction["amount"],
        "score": result["score"],
        "flag": flag,
        "latency_ms": round(elapsed_ms, 2),
        "timestamp": time.time()
    }
    r.lpush(RECENT_TXN_KEY, json.dumps(entry))
    r.ltrim(RECENT_TXN_KEY, 0, RECENT_TXN_MAX - 1)

    r.hincrby(STATS_KEY, "total_processed", 1)
    if flag == "FLAGGED":
        r.hincrby(STATS_KEY, "total_flagged", 1)
    elif flag == "UNKNOWN ACCOUNT":
        r.hincrby(STATS_KEY, "total_unknown", 1)
    r.hset(STATS_KEY, "last_latency_ms", round(elapsed_ms, 2))


def main():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    # Connect to Redis with retry
    print(f"Connecting to Redis at {redis_host}:{redis_port}...")
    while True:
        try:
            r = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                ssl=redis_ssl,
                decode_responses=True
            )
            r.ping()
            print("Connected to Redis successfully.")
            break
        except Exception as e:
            print(f"Waiting for Redis ({e}). Retrying in 2s...")
            time.sleep(2)

    model = load_model()
    print("Loaded trained XGBoost model.")

    # Connect to Kafka with retry
    print(f"Connecting to Kafka at {kafka_servers}...")
    while True:
        try:
            consumer = KafkaConsumer(
                'transactions',
                bootstrap_servers=kafka_servers,
                auto_offset_reset='latest',
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            print("Connected to Kafka topic 'transactions'.")
            break
        except Exception as e:
            print(f"Waiting for Kafka ({e}). Retrying in 2s...")
            time.sleep(2)

    print("Listening for transactions... (Ctrl+C to stop)\n")

    latencies = []
    FRAUD_THRESHOLD = 0.5

    for message in consumer:
        transaction = message.value

        start = time.perf_counter()
        result = score_transaction(r, model, transaction)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)

        if result["score"] is None:
            flag = "UNKNOWN ACCOUNT"
        elif result["score"] > FRAUD_THRESHOLD:
            flag = "FLAGGED"
        else:
            flag = "ok"

        print(
            f"Txn #{transaction['transaction_id']}: "
            f"{transaction['from_account']} -> {transaction['to_account']}, "
            f"score={result['score']}, [{flag}] "
            f"({elapsed_ms:.2f} ms)"
        )

        push_to_dashboard(r, transaction, result, elapsed_ms, flag)

        if len(latencies) % 20 == 0:
            avg = sum(latencies[-20:]) / 20
            print(f"  -- avg latency over last 20 txns: {avg:.2f} ms --")


if __name__ == "__main__":
    main()