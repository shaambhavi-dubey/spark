"""
Trains an XGBoost model that takes [sender_embedding + receiver_embedding + amount]
as input and predicts fraud.

FIXED VERSION: the first version trained GraphSAGE's loss on every node, then
reused those same nodes' embeddings to train and evaluate XGBoost -- meaning
XGBoost was partly just re-reading labels GraphSAGE's own training had already
baked into the embeddings. That gave a meaningless F1=1.000.

This version splits nodes into train/test BEFORE training GraphSAGE. GraphSAGE's
loss is computed ONLY on train nodes. Test nodes still get embeddings (via the
normal forward pass), but those embeddings were never directly shaped by their
own label during training -- so evaluating XGBoost on transactions into test
nodes is now a fair, non-circular test.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier


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


def main():
    print("Building graph...")
    graph = generate_synthetic_graph()

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

    # ---- KEY FIX: split nodes BEFORE training GraphSAGE ----
    n_nodes = data.num_nodes
    indices = np.arange(n_nodes)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=data.y.numpy(), random_state=42
    )
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    print(f"GraphSAGE will train its loss on {train_mask.sum().item()} nodes; "
          f"{n_nodes - train_mask.sum().item()} nodes are held out.")

    print("Training GraphSAGE (loss on train nodes only)...")
    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=64, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    class_counts = torch.bincount(data.y[train_mask])
    class_weights = torch.sqrt((class_counts.sum() / class_counts).float())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])  # ONLY train nodes affect the model
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index)
    embeddings = embeddings.numpy()
    print(f"Embeddings shape: {embeddings.shape} (computed for ALL nodes, "
          f"but held-out nodes never influenced GraphSAGE's weights)")

    # ---- Build edge-level dataset for XGBoost ----
    print("Building edge-level training data for XGBoost...")
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    rows, labels, receiver_is_train = [], [], []
    for u, v, edata in graph.edges(data=True):
        u_idx = id_to_idx[u]
        v_idx = id_to_idx[v]
        combined = np.concatenate([embeddings[u_idx], embeddings[v_idx], [edata['amount']]])
        rows.append(combined)
        labels.append(graph.nodes[v]['label'])
        receiver_is_train.append(bool(train_mask[v_idx]))

    X = np.array(rows)
    y = np.array(labels)
    receiver_is_train = np.array(receiver_is_train)

    # Edges are split by whether their RECEIVER was in GraphSAGE's train set.
    # Test edges point to nodes GraphSAGE's loss never touched -- a fair test.
    X_train, y_train = X[receiver_is_train], y[receiver_is_train]
    X_test, y_test = X[~receiver_is_train], y[~receiver_is_train]

    print(f"XGBoost train edges: {len(y_train)} (fraud: {y_train.sum()})")
    print(f"XGBoost test edges (receiver held out from GraphSAGE): {len(y_test)} (fraud: {y_test.sum()})")

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=4, scale_pos_weight=scale,
        eval_metric="aucpr", random_state=42
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"F1 (fair, held-out test): {f1_score(y_test, y_pred):.3f}")

    importances = xgb_model.feature_importances_
    print(f"Amount feature importance: {importances[-1]:.4f}")
    print(f"Sender embedding importance sum: {importances[:64].sum():.4f}")
    print(f"Receiver embedding importance sum: {importances[64:128].sum():.4f}")

    xgb_model.save_model("xgb_fraud_model.json")
    print("Saved trained model to xgb_fraud_model.json")


if __name__ == "__main__":
    main()