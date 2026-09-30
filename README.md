<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:161b22,100:1a6b3c&height=220&section=header&text=🛡️%20UPI%20FRAUD%20DETECTION&fontSize=42&fontColor=58a6ff&fontAlignY=35&desc=REAL-TIME%20PIPELINE&descSize=14&descColor=8b949e&descAlignY=55&animation=fadeIn" width="100%"/>

<div align="center">

<a href="#">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=80&lines=Graph+Neural+Networks+%2B+XGBoost+%2B+Kafka;Real-Time+Fraud+Scoring+in+%3C20ms;One+Command+to+Run+Everything" alt="Typing SVG" />
</a>

<br/>

<img src="https://img.shields.io/badge/Apache_Kafka-231F20?style=for-the-badge&logo=apache-kafka&logoColor=white" alt="Kafka"/>
<img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/XGBoost-189744?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>

<br/><br/>

<img src="https://img.shields.io/badge/⚡_Latency-3~20ms-58a6ff?style=flat-square" alt="Latency"/>
<img src="https://img.shields.io/badge/📊_Embeddings-64_dim-8b5cf6?style=flat-square" alt="Embeddings"/>
<img src="https://img.shields.io/badge/🧠_Model-GraphSAGE+XGBoost-1a6b3c?style=flat-square" alt="Model"/>
<img src="https://img.shields.io/badge/🔄_Retrain-Every_120s-e3b341?style=flat-square" alt="Retrain"/>
<img src="https://img.shields.io/badge/📜_License-MIT-30363d?style=flat-square" alt="License"/>

</div>

<br/>

<table>
<tr>
<td width="25%" align="center">
<h3>🕸️ Graph Intelligence</h3>
<sub>GraphSAGE on 10K+ node<br/>Barabási–Albert network<br/><i>neighborhood-aware embeddings</i></sub>
</td>
<td width="25%" align="center">
<h3>⚡ Real-Time Scoring</h3>
<sub>Kafka → Redis → XGBoost<br/>3–20ms per transaction<br/><i>sub-second fraud detection</i></sub>
</td>
<td width="25%" align="center">
<h3>🔄 Continuous Retraining</h3>
<sub>New accounts every 120s<br/>Auto-retrain GraphSAGE<br/><i>adapts to network growth</i></sub>
</td>
<td width="25%" align="center">
<h3>📊 Live Dashboard</h3>
<sub>Streamlit real-time UI<br/>fraud flags + latency charts<br/><i>zero-config monitoring</i></sub>
</td>
</tr>
</table>

<div align="center">

[Overview](#-overview) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[How It Works](#-how-fraud-detection-works) •
[Tech Stack](#-tech-stack) •
[Project Structure](#-project-structure) •
[Deployment](#-cloud-deployment) •
[Performance](#-performance)

</div>

---

## 🔍 Overview

A **production-style, end-to-end fraud detection system** for UPI (Unified Payments Interface) transactions. The pipeline combines **graph neural networks** (GraphSAGE) for learning account behavior from the transaction network with **XGBoost** for real-time fraud scoring — all orchestrated through **Kafka** and **Redis**, with a live **Streamlit** dashboard.

> **One command to run everything:** `docker compose up --build`

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────┐     ┌──────────────┐     ┌───────────┐
│  Retrain     │────▶│  Redis  │◀────│   Consumer   │◀────│   Kafka   │
│  (GraphSAGE) │     │         │     │  (XGBoost)   │     │           │
│              │     │         │     │              │     │           │
│  • Trains GNN│     │Embeddings│    │  • Scores    │     │           │
│  • Pushes    │     │& Stats  │     │    txns      │     │           │
│    embeddings│     │         │     │  • Pushes    │     │           │
│  • Writes    │     └────┬────┘     │    results   │     └─────▲─────┘
│    graph     │          │          └──────────────┘           │
└──────┬───────┘          │                                    │
       │            ┌─────▼─────┐    ┌──────────────┐          │
       │            │ Dashboard │    │   Producer   │──────────┘
       └───────────▶│(Streamlit)│    │              │
    shared_graph.pkl│           │    │  • Reads     │
                    │  Live UI  │    │    graph     │
                    │  :8501    │    │  • Sends     │
                    └───────────┘    │    txns      │
                                    └──────────────┘
```

### Pipeline Flow

1. **Retrain Loop** — Generates a synthetic transaction graph, trains a 2-layer GraphSAGE model, and pushes 64-dim node embeddings into Redis. Periodically adds new accounts and retrains (simulating network growth).

2. **Producer** — Reads the shared transaction graph and publishes synthetic transactions to a Kafka topic at ~2 txns/sec.

3. **Consumer** — Consumes transactions from Kafka, fetches sender/receiver embeddings from Redis, concatenates them with the transaction amount, and runs XGBoost inference. Flags transactions with fraud probability > 0.5.

4. **Dashboard** — Reads scoring results from Redis and displays live metrics, a transaction table with fraud highlighting, and latency charts.

---

## ⚡ Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Docker Compose v2)

### Run

```bash
git clone https://github.com/<your-username>/upi-fraud-detection.git
cd upi-fraud-detection
docker compose up --build
```

Open **[http://localhost:8501](http://localhost:8501)** to see the live dashboard.

> **Note:** First build takes 5–10 minutes (PyTorch + XGBoost downloads). Subsequent starts are instant.

### Stop

```bash
docker compose down        # Stop everything
docker compose down -v     # Stop + delete all data (full reset)
```

---

## 🧠 How Fraud Detection Works

### Graph Construction
- **10,000 legitimate accounts** connected via a Barabási–Albert graph (scale-free, mimics real payment networks)
- **250 mule/fraud rings** injected — each mule receives transfers from 15–40 source accounts with suspiciously uniform amounts

### Feature Engineering

| Feature | Description |
|---------|------------|
| `account_age_days` | How long the account has existed |
| `avg_amt` | Average transaction amount (in + out) |
| `max_amount` | Maximum single transaction |
| `in_degree` | Number of incoming connections |
| `out_degree` | Number of outgoing connections |

### GraphSAGE Embeddings
A 2-layer GraphSAGE model is trained on the graph with node features. The hidden layer produces **64-dimensional embeddings** that capture each account's neighborhood structure — mule accounts that receive many small, uniform transfers from diverse sources get distinctly different embeddings from legitimate accounts.

### Scoring
For each transaction `(sender → receiver, amount)`:
1. Fetch sender and receiver embeddings from Redis (64 dims each)
2. Concatenate: `[sender_emb | receiver_emb | amount]` → **129 features**
3. XGBoost predicts `P(fraud)` — threshold at **0.5**

### Continuous Retraining
Every 120 seconds, the retrain loop:
1. Adds **3,000 new accounts** to the graph
2. Retrains GraphSAGE (**300 epochs**)
3. Pushes updated embeddings to Redis
4. Writes the updated graph for the producer to pick up

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Message Broker** | Apache Kafka (KRaft) | Transaction stream between producer ↔ consumer |
| **Embedding Store** | Redis | GraphSAGE embeddings + dashboard stats |
| **Graph ML** | PyTorch + PyTorch Geometric | GraphSAGE (2-layer, 64-dim embeddings) |
| **Fraud Classifier** | XGBoost | Binary classification on concatenated embeddings |
| **Dashboard** | Streamlit | Live metrics, transaction table, latency charts |
| **Orchestration** | Docker Compose | All services with healthchecks and dependency ordering |

---

## 📁 Project Structure

```
├── producer/
│   └── produce_transactions.py    # Kafka producer — synthetic transactions
├── consumer/
│   └── score_transactions.py      # Kafka consumer — XGBoost scoring
├── retrain/
│   └── periodic_retrain.py        # GraphSAGE training loop
├── embeddings/
│   ├── xgb_on_embeddings.py       # XGBoost training on embeddings
│   └── train and push.py          # One-shot train + push to Redis
├── app.py                         # Streamlit dashboard
├── docker/
│   ├── Dockerfile.producer
│   ├── Dockerfile.consumer
│   ├── Dockerfile.retrain
│   ├── Dockerfile.dashboard
│   └── requirements-*.txt         # Per-service dependencies
├── docker-compose.yml             # Full pipeline orchestration
├── xgb_fraud_model.json           # Pre-trained XGBoost model
└── shared_graph.pkl               # Seed transaction graph
```

---

## ☁️ Cloud Deployment

The full pipeline (Kafka + Redis + 4 services) requires a VM. But you can deploy **just the dashboard** to [Streamlit Community Cloud](https://streamlit.io/cloud) for free:

1. Push this repo to GitHub
2. Get a free Redis instance from [Upstash](https://upstash.com/) or [Redis Cloud](https://redis.com/try-free/)
3. Run the producer + consumer locally (they push data to your cloud Redis)
4. Deploy `app.py` on Streamlit Cloud with these secrets:

```toml
REDIS_HOST = "your-redis-host.upstash.io"
REDIS_PORT = 6379
REDIS_PASSWORD = "your_password"
REDIS_SSL = "true"
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Scoring latency | **~3–20 ms** per transaction |
| Throughput | ~2 txns/sec (configurable) |
| Retrain cycle | ~30s for 10K+ nodes |
| Embedding dimensions | 64 |
| Model | XGBoost (binary classifier) |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:161b22,100:1a6b3c&height=120&section=footer" width="100%"/>
