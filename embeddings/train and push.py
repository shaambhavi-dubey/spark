"""
Phase 2: Train GraphSAGE on the synthetic UPI graph, and push every account's
embedding into Redis, keyed by account ID.

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


# ---- Step 1: reuse the synthetic graph generator (same as gnn-upi notebook 07) ----

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


# ---- Step 2: GraphSAGE model (same architecture as notebook 05) ----

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        embedding = h  # this is the embedding: the representation BEFORE the final layer
        out = self.conv2(h, edge_index)
        return out, embedding


def main():
    print("Generating synthetic graph...")
    graph = generate_synthetic_graph()

    print("Building features...")
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

    node_ids = list(graph.nodes())  # preserves the same order from_networkx used

    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)

    print("Training GraphSAGE...")
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
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Extracting embeddings...")
    model.eval()
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index)
    embeddings = embeddings.numpy()

    print(f"Embeddings shape: {embeddings.shape}")

    print("Pushing embeddings to Redis...")
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    for i, node_id in enumerate(node_ids):
        embedding_list = embeddings[i].tolist()
        r.set(f"embedding:{node_id}", json.dumps(embedding_list))

    print(f"Pushed {len(node_ids)} embeddings to Redis.")

    # quick sanity check: look up one embedding
    sample_id = node_ids[0]
    stored = json.loads(r.get(f"embedding:{sample_id}"))
    print(f"Sample check -- account {sample_id} embedding (first 5 values): {stored[:5]}")


if __name__ == "__main__":
    main()