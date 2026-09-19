# ArXiv RAG Engine: Autonomous Research Assistant

---

## Project Overview

The **ArXiv RAG Engine** is a Retrieval-Augmented Generation system that provides grounded, citation-backed answers to scientific research questions. It combines **high-performance C++ text processing**, **fine-tuned neural embeddings**, and a **ReAct-style agent loop** to retrieve and synthesize knowledge from ArXiv ML papers.

The engine learns a specialized semantic space where papers are clustered by conceptual similarity rather than shared vocabulary. When initial retrieval is insufficient, the agent autonomously reformulates its search query and retries — producing more accurate, evidence-grounded responses.

---

## Key Features

### 🚀 High-Performance Architecture
- **C++ Accelerated Preprocessing:** Text normalization and TF-IDF keyword extraction implemented in **C++17** and exposed to Python via **PyBind11**, eliminating interpreter-level bottlenecks for large corpus processing.
- **PostgreSQL + pgvector:** Industry-standard vector storage with **IVFFlat indexing** (dynamic `lists = √N`, `probes = √lists`) for fast approximate nearest-neighbor retrieval.
- **Efficient Ingestion:** PostgreSQL `COPY FROM STDIN` protocol for high-throughput bulk data loading.

### 🧠 Representation Learning
- **Base Encoder:** `allenai-specter` — a transformer pretrained on scientific paper titles, abstracts, and citation graphs.
- **Metric Learning:** Custom **PyTorch projection head** fine-tuned with **Triplet Loss**, compressing 768-dim SPECTER embeddings into a learned 256-dim semantic space.
- **IDF-Weighted Similarity:** Positive and negative training pairs are defined by corpus-level IDF keyword overlap scores — rare shared terms contribute more signal than common ones.
- **Hyperparameter Optimization:** Grid search over `lr`, `margin`, and `projection_dim` via **W&B Sweeps** across 36 runs with an 80/10/10 train/val/test split. The final projection head achieves **48.8% Recall@5** on the held-out test set.

### 🤖 Autonomous RAG Pipeline
- **Gemini-Powered Agent:** Integrated with the **Google Gemini API** using a **ReAct-style tool-calling loop** — the model decides whether retrieved papers are sufficient or whether to refine its query and search again.
- **Grounded Generation:** Strict adherence to retrieved context with mandatory numerical citations (`[1]`, `[2]`), reducing hallucinations.
- **Interactive CLI:** Session management and slash-commands (`/k`, `/sources`, `/help`, `/exit`).

---

## Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+, C++17 |
| **Machine Learning** | PyTorch, Sentence-Transformers, NumPy |
| **LLM / Agent** | Google Gemini API (GenAI SDK) |
| **Database** | PostgreSQL, pgvector |
| **C++ Bindings** | PyBind11, CMake |
| **Experiment Tracking** | Weights & Biases (W&B) |
| **Infrastructure** | Docker, Docker Compose |

---

## System Architecture

The pipeline runs in five sequential phases:

1. **Ingestion:** Streaming arXiv papers from HuggingFace datasets, lexical analysis via C++ TF-IDF extractor.
2. **Normalization:** Stop-word removal and text cleaning via C++ `TextNormalizer`.
3. **Representation:** 768-dim SPECTER base embeddings → fine-tuned 256-dim projection head.
4. **Indexing:** IVFFlat k-means clustering in pgvector for approximate nearest-neighbor search.
5. **Autonomous RAG:** ReAct agent loop — retrieve → evaluate → refine query or generate answer.

---

## Deep Dive: Metric Learning

The projection head is trained using **Triplet Loss**:

$$L = \max(0,\ d(a, p) - d(a, n) + \text{margin})$$

- **Anchor (a):** A query paper.
- **Positive (p):** A paper with high IDF-weighted keyword overlap — shares rare, domain-specific terminology.
- **Negative (n):** A paper from the same broad category but with low keyword overlap.

Triplet mining follows a three-tier hierarchy: **semi-hard → hard → easy**, prioritizing the most informative training signal at each step. The threshold separating positives from negatives is set at `1.75 × avg_idf_score`, derived empirically by measuring the coverage-recall tradeoff across the validation set.

---

## Getting Started

### 1. Build the Application Image

Compile the custom C++ extensions (`normalizer`, `tfidf`), download Python packages, and build the runtime Docker image:

```bash
docker compose build app
```

### Prerequisites
- Docker Desktop installed and running
- Create a `.env` file in the root directory:
  ```env
  DB_USER=postgres
  DB_PASSWORD=yourpassword
  DB_NAME=arxiv_engine
  GEMINI_API_KEY=your_gemini_key

> **Why build first?** Building compiles the C++ libraries into `.so` shared objects for the container's Linux architecture and installs heavy dependencies like PyTorch. Run this once on first setup, and again whenever you modify C++ source files, `requirements.txt`, or the `Dockerfile`.

---

### 2. Data Pipeline & Database Initialization

Initialize the PostgreSQL schema, fetch arXiv papers, generate embeddings, and build IVFFlat indices:

```bash
docker compose run --rm app bash scripts/init_db.sh
```

> **What this does:** Overrides the container's default startup command and runs the pipeline script inside an ephemeral container. It checks for pre-existing records and trained weights, skips redundant steps, and populates the database end-to-end.

---

### 3. Launch the Assistant

Start the interactive CLI session:

```bash
docker compose run --rm app
```

> **Why no trailing command?** Docker executes the default `CMD ["python", "-m", "src.main"]` from the `Dockerfile`, launching directly into the assistant prompt.

---

### 4. Lifecycle Management & Teardown

#### Pausing — `docker compose stop`
```bash
docker compose stop
```
Halts containers without removing them. Resume with `docker compose start`.

#### Ending a Session — `docker compose down`
```bash
docker compose down
```
Removes containers and networks. PostgreSQL data and model weights remain on disk.

#### Full Reset — `docker compose down -v`
```bash
docker compose down -v
```
Removes containers, networks, and volumes. Permanently deletes all stored papers, embeddings, and index tables.

---

### Interactive Commands

| Command | Description |
|---|---|
| `/k <n>` | Adjust the number of retrieved papers (e.g., `/k 10`) |
| `/sources` | Inspect full metadata and abstracts of papers used in the last answer |
| `/clear` | Clear the terminal screen |
| `/help` | View the command list |
| `/exit` | Gracefully shut down the engine and connection pools |