# Semantic Movie Search Engine

---

# Project Overview

This project implements a high-performance semantic movie search engine that moves beyond traditional keyword matching. Instead of relying solely on lexical similarity, it learns a semantic embedding space where distance represents thematic similarity between movies.

The system uses a pretrained **SentenceTransformer (`all-MiniLM-L6-v2`)** together with a custom **PyTorch projection head** trained using **Triplet Loss**. The resulting model compresses movie representations into a **128-dimensional embedding space**, allowing semantically similar films to cluster together.

To maximize performance, text preprocessing is implemented in **C++** and exposed to Python through **PyBind11**, eliminating Python preprocessing bottlenecks during embedding generation.

Movie embeddings are stored inside **PostgreSQL** using the **pgvector** extension, while an **IVFFlat** index enables extremely fast nearest-neighbor search.

---

# Features

- Semantic movie search using dense vector embeddings
- Fine-tuned projection head trained with Triplet Loss
- High-performance C++ text preprocessing with PyBind11
- PostgreSQL + pgvector vector database
- IVFFlat approximate nearest-neighbor indexing
- Dockerized infrastructure
- Unit testing for both Python and C++ components

---

# Tech Stack

## Machine Learning

- PyTorch
- SentenceTransformers (HuggingFace)
- NumPy

## Data Engineering

- C++
- PyBind11
- Pandas

## Database

- PostgreSQL
- pgvector
- psycopg_pool (`COPY FROM STDIN` streaming)

## Infrastructure

- Docker
- Docker Compose

## Testing

- PyTest
- Google Test

---

# Architecture

The project is divided into three main stages:

1. Data preprocessing
2. Model training
3. Vector database generation and search

---

# High-Performance Text Processing

## TextNormalizer (C++ / PyBind11)

Large datasets can make text preprocessing a bottleneck. Instead of cleaning text in Python, preprocessing is implemented in C++.

The normalizer performs the following operations sequentially:

- Removes ASCII punctuation
- Converts text to lowercase
- Removes stopwords using an `unordered_set` for **O(1)** lookup

This significantly reduces preprocessing overhead before embedding generation.

---

# Embedding Models

## MovieEmbedder

The baseline embedding model uses the pretrained **all-MiniLM-L6-v2** SentenceTransformer.

Characteristics:

- Output dimension: **384**
- Batch size: **32**
- Frozen pretrained encoder
- Used as the baseline semantic search model

---

## FilmEncoder

The fine-tunable model extends the pretrained encoder.

Instead of retraining the entire transformer, the encoder is frozen and a lightweight projection head is attached.

Architecture:

```
384-d embedding
        │
     Dropout
    (p = 0.25)
        │
 Linear Projection
        │
128-d embedding
        │
 L2 Normalization
```

The final normalization allows cosine similarity to be computed efficiently using a dot product.

---

# Triplet Loss

The projection head is optimized using **Triplet Loss** with a margin of **0.2**.

The objective is

```
distance(anchor, positive)
<
distance(anchor, negative) - margin
```

which forces semantically similar movies to become closer together while pushing unrelated movies farther apart.

---

# Triplet Mining

## FilmPairDataset

The dataset constructs triplets using movie metadata.

Every pair of movies is assigned the following similarity score:

```text
Score = (Shared Genres × 1.5) + (Shared Keywords × 1.0)
```

These scores are used to generate positive and negative candidate pools.

Negative sampling follows the hierarchy:

1. Semi-hard negatives
2. Hard negatives
3. Easy negatives

Semi-hard negatives provide the strongest learning signal and are therefore preferred whenever available.

---

## custom_collate_fn

PyTorch's default DataLoader collate function cannot handle invalid triplets.

The custom collate function safely removes samples where:

- no positive candidate exists
- no negative candidate exists

This prevents training crashes while keeping batch generation efficient.

---

# Experiments

Developing the metric learning model required multiple experiments to understand how triplet sampling affects embedding geometry.

---

## Experiment 1 — Distance Matrix Feedback Loop

### Hypothesis

Recompute the pairwise distance matrix every three epochs so semi-hard negative mining uses the newest embeddings.

### Result

Failure

### Analysis

The updated distance matrix was generated using an almost untrained projection head.

Early in training, these embeddings behave almost randomly.

Rebuilding neighborhoods from unstable vectors caused the model to optimize against its own noise rather than stable metadata relationships, resulting in catastrophic collapse.

### Conclusion

Distance matrices should only be generated from stable embeddings, or triplet mining should be performed online.

---

## Experiment 2 — Thresholding and Franchise Collapse

### Hypothesis

Restrict positive pairs using:

- Score ≥ 6
- Maximum 15 positive candidates

### Result

Good

### Analysis

Movies from the same franchise monopolized each other's candidate lists.

For example:

- Batman Begins
- The Dark Knight
- Batman Returns

became isolated clusters because they were trained almost exclusively against one another.

The resulting embedding space contained extremely dense franchise clusters with poorly defined global boundaries.

Search quality deteriorated despite apparently successful training.

---

## Experiment 3 — The Gray Zone

### Hypothesis

Negative examples should share zero genres.

### Result

Failure

### Analysis

Movies sharing even one genre were excluded from both positive and negative pools.

This created a large "gray zone" where many informative samples were never used during training.

The model therefore failed to learn subtle distinctions such as:

- superhero action
- generic action

because generic action films never appeared as meaningful negatives.

### Conclusion

Relaxing the negative definition significantly improves representation learning.

---

# Results

The experiments revealed an important distinction between pretrained and fine-tuned embeddings.

## Where the pretrained model excels

The pretrained **all-MiniLM-L6-v2** performs extremely well when lexical similarity is already high.

For example:

```
The Matrix
↓
The Matrix Reloaded
↓
The Matrix Revolutions
```

Movie descriptions naturally contain the same characters, locations and terminology, making lexical similarity sufficient.

---

## Where the fine-tuned model excels

The projection head performs much better for abstract thematic similarity.

Example:

```
Inception
```

The pretrained model often retrieves generic action or sci-fi films because their overviews share similar vocabulary.

The fine-tuned model instead learns to emphasize metadata relationships such as:

- subconscious
- dreams
- reality
- mind

As a result, movies like:

- Looper
- The Cell

become much closer in the learned embedding space despite sharing relatively little vocabulary.

The fine-tuned model successfully captures thematic similarity rather than surface-level text similarity.

---

# Installation

## 1. Start PostgreSQL

```bash
docker-compose up -d
```

---

## 2. Build the C++ normalizer

```bash
cd normalizer

mkdir build
cd build

cmake ..
make
```

---

## 3. Initialize the database

Creates the schema, enables pgvector, and loads the cleaned dataset.

```bash
python -m src.model.setup
```

---

## 4. Train the projection head

```bash
python -m src.model.train
```

---

## 5. Generate embeddings and build the IVFFlat index

```bash
python -m src.model.main --model finetuned
```
