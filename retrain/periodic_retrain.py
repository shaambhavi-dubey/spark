"""
Phase 5 (wired to producer): Offline periodic retraining loop.

Retrains GraphSAGE on the current graph, refreshes Redis embeddings, AND
writes the current graph out to a shared file that the producer reloads
periodically -- so new accounts added here actually show up as real
transactions the consumer gets tested against, closing the gap between the
two processes.
"""

import networkx as nx
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
import redis
import json
import time
import datetime
import pickle
import os

SHARED_GRAPH_PATH = os.getenv("SHARED_GRAPH_PATH", "shared_graph.pkl")


def generate_synthetic_graph(n_legit=10000, n_rings=250,
                              sources_range=(15, 40),
                              fraud_amt_range=None,
                              seed=42):
    random.seed(seed)
    np.random.seed(seed)
    Gr = nx.barabasi_albert_graph(n=n_legit, m=3, seed=seed)
    DirGr = nx.DiGraph()
    DirGr.add_nodes_from(Gr.nodes())
    for u, v in Gr.edges():
        if random.random() < 0.5:
            DirGr.add_edge(u, v)
        else:
            DirGr.add_edge(v, u)
    for u, v in DirGr.edges():
        amount = np.random.lognormal(mean=4, sigma=1.2)
        DirGr[u][v]['amount'] = round(amount, 2)
    account_age_days = {n: random.randint(30, 2000) for n in DirGr.nodes()}
    nx.set_node_attributes(DirGr, account_age_days, 'account_age_days')
    nx.set_node_attributes(DirGr, 0, 'label')
    all_nodes = list(DirGr.nodes())
    random.shuffle(all_nodes)
    mule_nodes = all_nodes[:n_rings]
    remaining_pool = all_nodes[n_rings:]
    for mule in mule_nodes:
        n_sources = random.randint(*sources_range)
        sources = random.sample(remaining_pool, n_sources)
        if fraud_amt_range == "camouflaged":
            for src in sources:
                if src == mule:
                    continue
                amt = round(np.random.lognormal(mean=4, sigma=1.2), 2)
                DirGr.add_edge(src, mule, amount=amt)
        else:
            min_amt = random.uniform(5, 20)
            max_amt = min_amt + random.uniform(50, 200)
            for src in sources:
                if src == mule:
                    continue
                amt = round(random.uniform(min_amt, max_amt), 2)
                DirGr.add_edge(src, mule, amount=amt)
        DirGr.nodes[mule]['label'] = 1
    return DirGr


def add_new_accounts(graph, n_new=50, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    existing_nodes = list(graph.nodes())
    max_id = max(existing_nodes)
    new_ids = list(range(max_id + 1, max_id + 1 + n_new))

    for new_id in new_ids:
        graph.add_node(new_id)
        graph.nodes[new_id]['account_age_days'] = random.randint(0, 5)
        graph.nodes[new_id]['label'] = 0

        n_connections = random.randint(1, 4)
        partners = random.sample(existing_nodes, n_connections)
        for p in partners:
            amount = round(np.random.lognormal(mean=4, sigma=1.2), 2)
            if random.random() < 0.5:
                graph.add_edge(new_id, p, amount=amount)
            else:
                graph.add_edge(p, new_id, amount=amount)

    return graph, new_ids


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        embedding = h
        out = self.conv2(h, edge_index)
        return out, embedding


def save_shared_graph(graph):
    """
    Writes the graph to disk so the producer can pick it up. Writes to a
    temp file first, then atomically renames it -- this guarantees the
    producer never reads a half-written, corrupted file, since os.replace
    is atomic at the OS level.
    """
    target_dir = os.path.dirname(SHARED_GRAPH_PATH)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    temp_path = SHARED_GRAPH_PATH + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(graph, f)
    os.replace(temp_path, SHARED_GRAPH_PATH)


def retrain_and_push(graph, r):
    print(f"[{datetime.datetime.now().isoformat()}] Starting retrain on "
          f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges...")

    for n in graph.nodes():
        incoming = [graph[u][n]['amount'] for u in graph.predecessors(n)]
        outgoing = [graph[n][t]['amount'] for t in graph.successors(n)]
        total = incoming + outgoing
        graph.nodes[n]['avg_amt'] = sum(total) / len(total) if total else 0.0
        graph.nodes[n]['max_amount'] = float(max(total)) if total else 0.0
        graph.nodes[n]['in_degree'] = graph.in_degree(n)
        graph.nodes[n]['out_degree'] = graph.out_degree(n)

    data = from_networkx(
        graph,
        group_node_attrs=['account_age_days', 'avg_amt', 'max_amount', 'in_degree', 'out_degree']
    )
    data.y = torch.tensor([graph.nodes[n]['label'] for n in graph.nodes()], dtype=torch.long)
    node_ids = list(graph.nodes())

    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)

    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=64, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    class_counts = torch.bincount(data.y)
    class_weights = torch.sqrt((class_counts.sum() / class_counts).float())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index)
    embeddings = embeddings.numpy()

    pipe = r.pipeline(transaction=False)
    for i, node_id in enumerate(node_ids):
        pipe.set(f"embedding:{node_id}", json.dumps(embeddings[i].tolist()))
    pipe.execute()

    print(f"[{datetime.datetime.now().isoformat()}] Retrain complete. "
          f"Pushed {len(node_ids)} embeddings to Redis.")

    save_shared_graph(graph)
    print(f"Wrote updated graph to {SHARED_GRAPH_PATH} for the producer to pick up.\n")


def main(retrain_interval_seconds=None, new_accounts_per_cycle=None):
    if retrain_interval_seconds is None:
        retrain_interval_seconds = int(os.getenv("RETRAIN_INTERVAL_SECONDS", 120))
    if new_accounts_per_cycle is None:
        new_accounts_per_cycle = int(os.getenv("NEW_ACCOUNTS_PER_CYCLE", 3000))

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

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

    # Seed shared graph if seed exists and target does not exist yet
    if not os.path.exists(SHARED_GRAPH_PATH) and os.path.exists("shared_graph.pkl"):
        try:
            import shutil
            target_dir = os.path.dirname(SHARED_GRAPH_PATH)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            shutil.copyfile("shared_graph.pkl", SHARED_GRAPH_PATH)
            print(f"Initialized {SHARED_GRAPH_PATH} from local seed file.")
        except Exception as e:
            print(f"Seed note: {e}")

    print("Building initial graph...")
    graph = generate_synthetic_graph()

    # write it immediately so the producer has something to load on its own startup
    save_shared_graph(graph)

    cycle = 0
    while True:
        if cycle > 0:
            print(f"Simulating {new_accounts_per_cycle} new accounts joining the network...")
            graph, new_ids = add_new_accounts(graph, n_new=new_accounts_per_cycle, seed=cycle)
            print(f"New account IDs: {new_ids[0]}...{new_ids[-1]}")

        retrain_and_push(graph, r)

        cycle += 1
        print(f"Sleeping {retrain_interval_seconds} seconds until next retrain...\n")
        time.sleep(retrain_interval_seconds)


if __name__ == "__main__":
    main()