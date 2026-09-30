"""
Phase 3 (wired to retrain loop): Kafka producer.

Now loads its graph from the shared file written by retrain/periodic_retrain.py
instead of generating its own separate copy. Periodically checks if that file
has been updated (new accounts added by the retrain loop) and reloads if so --
this is what lets a genuinely new account show up in a live transaction and
get scored correctly by the consumer.

Run retrain/periodic_retrain.py FIRST (even just to let it write the initial
shared graph file), then start this.
"""

import random
import numpy as np
import json
import time
import os
import pickle
from kafka import KafkaProducer

SHARED_GRAPH_PATH = os.getenv("SHARED_GRAPH_PATH", "shared_graph.pkl")
RELOAD_CHECK_EVERY_N_TXNS = 10  # check for graph updates this often


def load_shared_graph():
    for _ in range(5):
        try:
            with open(SHARED_GRAPH_PATH, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, FileNotFoundError):
            time.sleep(0.5)
    with open(SHARED_GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def wait_for_shared_graph():
    print(f"Waiting for {SHARED_GRAPH_PATH} to exist "
          f"(start retrain/periodic_retrain.py if you haven't)...")
    if not os.path.exists(SHARED_GRAPH_PATH) and os.path.exists("shared_graph.pkl"):
        try:
            import shutil
            target_dir = os.path.dirname(SHARED_GRAPH_PATH)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            shutil.copyfile("shared_graph.pkl", SHARED_GRAPH_PATH)
            print(f"Initialized {SHARED_GRAPH_PATH} from local fallback.")
        except Exception as e:
            print(f"Fallback copy note: {e}")

    while not os.path.exists(SHARED_GRAPH_PATH):
        time.sleep(1)
    # tiny extra wait in case the file is still being written on first creation
    time.sleep(1)


def main():
    wait_for_shared_graph()

    graph = load_shared_graph()
    last_mtime = os.path.getmtime(SHARED_GRAPH_PATH)
    edges = list(graph.edges(data=True))
    print(f"Loaded shared graph: {graph.number_of_nodes()} nodes, {len(edges)} edges")

    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    print(f"Connecting to Kafka at {kafka_servers}...")
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Connected to Kafka producer successfully.")
            break
        except Exception as e:
            print(f"Waiting for Kafka ({e}). Retrying in 2s...")
            time.sleep(2)

    print("Publishing transactions to Kafka topic 'transactions'... (Ctrl+C to stop)\n")

    count = 0
    while True:
        # periodically check if the retrain loop has updated the shared graph
        if count % RELOAD_CHECK_EVERY_N_TXNS == 0:
            try:
                current_mtime = os.path.getmtime(SHARED_GRAPH_PATH)
                if current_mtime != last_mtime:
                    graph = load_shared_graph()
                    edges = list(graph.edges(data=True))
                    last_mtime = current_mtime
                    print(f">>> Reloaded updated graph: {graph.number_of_nodes()} nodes, "
                          f"{len(edges)} edges <<<")
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                # file was mid-write at the exact moment we checked -- skip this
                # check, we'll catch the update on the next cycle
                pass

        u, v, data = random.choice(edges)

        transaction = {
            "transaction_id": count,
            "from_account": u,
            "to_account": v,
            "amount": data["amount"],
            "timestamp": time.time(),
            "true_label": graph.nodes[v].get("label", 0)
        }

        producer.send('transactions', transaction)
        print(f"Sent txn #{count}: {u} -> {v}, amount={transaction['amount']}")

        count += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()