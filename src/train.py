import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import wandb
import os

from src.model.encoder import PaperEncoder
from src.data.dataset import PaperPairDataset, custom_collate_fn
from src.data.loader import (
    build_distance_matrix,
    fetch_idf_scores,
    fetch_paper_keywords,
    fetch_papers_embeddings
)
from src.model.loss import TripletLoss
from src.db import connection


GLOBAL_TOP_MODELS = []
test_dataloader = None
all_ids = None
all_embeddings = None
dataset = None


def train() -> None:
    """
    Executes a single hyperparameter trial managed by the W&B agent.

    Initializes connection pools, loads raw embeddings, lexical IDF statistics, and pairwise distances.
    Builds the PaperPairDataset, partitions 80/10/10 splits, and trains a PaperEncoder projection head.
    At each epoch, it computes train/val triplet losses and calculates validation Recall@5 by projecting
    the corpus and retrieving top candidates while dropping anchor self-matches.

    Maintains local early stopping and saves candidate checkpoints upon reaching a new best Recall@5.
    Upon run completion, it updates the global top-3 candidates sorted by:
    1. Primary: Recall@5 (descending)
    2. Secondary: Projection dimension (ascending)
    3. Tertiary: Triplet margin (descending)
    4. Quaternary: Best validation loss (ascending)

    Prunes non-qualifying checkpoint files from disk to manage local storage.

    Args:
        None

    Returns:
        None
    """
    wandb.init()
    config = wandb.config
    run_weights_path = f"./src/model/data/weights_{wandb.run.id}.pt"

    connection.init_pool()

    with connection.pool.connection() as conn:
        embeddings: dict[str, list[float]] = fetch_papers_embeddings(conn=conn)
        distance_matrix: dict[str, dict[str, float]] = build_distance_matrix(conn=conn)
        idf_scores: dict[str, float] = fetch_idf_scores(conn=conn)
        keywords: list[list[str]] = fetch_paper_keywords(conn=conn)

    avg_idf_score = sum(idf_scores.values()) / len(idf_scores)

    global dataset
    dataset = PaperPairDataset(
        matrix_distances=distance_matrix,
        embeddings=embeddings,
        idf_scores=idf_scores,
        avg_idf_score=avg_idf_score,
        keywords=keywords,
        margin=config.margin,
    )

    # calculate index boundaries
    train_end = int(0.8 * len(dataset))
    val_start = int(0.9 * len(dataset))

    # split the datasets
    training_dataset = Subset(dataset=dataset, indices=range(0, train_end))
    test_dataset = Subset(dataset=dataset, indices=range(train_end, val_start))
    validation_dataset = Subset(dataset=dataset, indices=range(val_start, len(dataset)))

    # update DataLoaders
    training_dataloader = DataLoader(
        dataset=training_dataset,
        shuffle=True,
        batch_size=32,
        collate_fn=custom_collate_fn,
    )
    
    # the dataset for the final evaluation among top3 models based on val dataset
    global test_dataloader 
    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=16,
        collate_fn=custom_collate_fn,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        batch_size=16,
        collate_fn=custom_collate_fn,
    )

    # map all indices to real papers' ids and get embedding for every paper
    global all_ids, all_embeddings
    all_ids = [dataset.get_paper_id(idx) for idx in range(len(dataset))]
    all_embeddings = [dataset.get_embedding(pid) for pid in all_ids]

    # diagnostics for positive vectors 
    val_ids = list(validation_dataset.indices)
    pos_counts = [len(dataset.get_positives(dataset.get_paper_id(idx))) for idx in val_ids]
    valid_queries = sum(1 for c in pos_counts if c > 0)
    print(f"Validation papers with >= 1 positive: {valid_queries} / {len(val_ids)}")
    print(f"Theoretical maximum Recall@5: {valid_queries / len(val_ids):.4f}")
    print(f"Average positives per query: {sum(pos_counts) / len(pos_counts):.2f}")

    model = PaperEncoder(projection_dim=config.projection_dim)
    optimizer = AdamW(params=model.parameters(), lr=config.lr)
    triplet_loss = TripletLoss(margin=config.margin)

    run_best_recall = 0.0
    run_best_val_loss = float('inf')
    patience = 0

    for epoch in range(config.epochs):
        # training mode - Dropout is on 
        model.train()
        total_train_loss = 0

        for anchors, positives, negatives in training_dataloader:
            if anchors.size(0) == 0:
                continue

            optimizer.zero_grad()

            # load tensors to GPU/CUDA or CPU as a fallback
            anchors = anchors.to(model._device)
            positives = positives.to(model._device)
            negatives = negatives.to(model._device)

            # triplet loss = max(0, d(anc, pos) - d(anc, neg) - margin)
            loss = triplet_loss(model(anchors), model(positives), model(negatives))
            loss.backward()     # compute grads
            optimizer.step()

            total_train_loss += loss.item()

        mean_train_loss = total_train_loss / len(training_dataloader)

        # validation mode - no Dropouts 
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for anchors, positives, negatives in validation_dataloader:
                if anchors.size(0) == 0:
                    continue

                anchors = anchors.to(model._device)
                positives = positives.to(model._device)
                negatives = negatives.to(model._device)

                loss = triplet_loss(model(anchors), model(positives), model(negatives))
                total_val_loss += loss.item()

            # project every embedding with a new head
            all_tensor = torch.tensor(all_embeddings).to(model._device)
            projected_corpus = model(all_tensor)

            # get projected embeddings for validation papers
            val_ids = list(validation_dataset.indices)
            val_queries = projected_corpus[val_ids]

            # normalize all vectors before mul 
            projected_corpus = torch.nn.functional.normalize(projected_corpus, p=2, dim=1)
            val_queries = torch.nn.functional.normalize(val_queries, p=2, dim=1)

            # compute cosine similarity between every validation vector and all vectors
            similarity_matrix = val_queries @ projected_corpus.T    # shape 400 X 4000
            # get top 10 since with the highest score
            _, top_k = torch.topk(similarity_matrix, k=10, dim=1, largest=True)

            hits = 0
            evaluable_queries = 0       # num of queries that have positive vectors

            for row_idx, val_idx in enumerate(val_ids):
                anchor_id = dataset.get_paper_id(val_idx)

                # get all positives based on TF-IDF score from dataset for curr anchor
                real_positives = dataset.get_positives(anchor_id)
                if not real_positives:
                    continue

                evaluable_queries += 1

                # filter out anchor itself and grab top 5 candidates
                predicted_ids = [all_ids[c] for c in top_k[row_idx] if all_ids[c] != anchor_id][:5]

               # if at least one real positive appeared in best top 5 after projection - we have a success 
                for pos_id in real_positives:
                    if pos_id in predicted_ids:
                        hits += 1
                        break
                

        mean_val_loss = total_val_loss / len(validation_dataloader)
        recall_at_5 = hits / evaluable_queries if evaluable_queries > 0 else 0.0

        print(f"Epoch {epoch + 1} — train: {mean_train_loss:.4f}, val: {mean_val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "recall_at_5": recall_at_5
        })

        # early stopping + save best
        if recall_at_5 > run_best_recall:
            run_best_recall = recall_at_5
            run_best_val_loss = mean_val_loss
            patience = 0
            wandb.log({"best_recall": run_best_recall})
            model.save(path=run_weights_path)
        else:
            patience += 1
            if patience >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    global GLOBAL_TOP_MODELS

    GLOBAL_TOP_MODELS.append({
        "run_id": wandb.run.id,
        "recall": run_best_recall,
        "val_loss": run_best_val_loss,
        "dim": config.projection_dim,
        "margin": config.margin,
        "path": run_weights_path
    })
    
    # Sort: Primary=Recall (desc), Secondary=Dim (asc), Tertiary=Margin (desc), Quaternary=Val Loss (asc)
    GLOBAL_TOP_MODELS = sorted(
        GLOBAL_TOP_MODELS, 
        key=lambda x: (-x["recall"], x["dim"], -x["margin"], x["val_loss"])
    )
    
    # identify the models that got kicked out of the Top 3
    losers = GLOBAL_TOP_MODELS[3:]
    GLOBAL_TOP_MODELS = GLOBAL_TOP_MODELS[:3]  # keep only the top 3
    
    # delete the losers' weights from the hard drive
    for loser in losers:
        if os.path.exists(loser["path"]):
            os.remove(loser["path"])

    connection.close_pool()
    wandb.finish()


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {"name": "recall_at_5", "goal": "maximize"},
        "parameters": {
            "lr":             {"values": [1e-4, 5e-4, 8e-4]},
            "projection_dim": {"values": [128, 256, 384]},
            "margin":         {"values": [0.15, 0.2, 0.25, 0.3]},
            "epochs":         {"value": 30},
            "patience":       {"value": 5},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="arxiv-research-assistant")
    wandb.agent(sweep_id, function=train)

    print("\n--- Running Final Evaluation on Test Set ---")
    best_test_recall = -1.0
    winning_model_path = None

    # test_ids covers 80% to 90%
    test_ids = list(range(int(0.8 * len(all_ids)), int(0.9 * len(all_ids))))
    test_tensor = torch.tensor([all_embeddings[idx] for idx in test_ids])
    corpus_tensor = torch.tensor(all_embeddings)

    for entry in GLOBAL_TOP_MODELS:
        # instantiate the model with the exact dimensions from the sweep config
        model = PaperEncoder(projection_dim=entry["dim"])
        
        # load the state dict manually
        checkpoint = torch.load(entry["path"], map_location=model._device)
        model._projection.load_state_dict(checkpoint["projection_state_dict"])
        
        model.eval()

        test_tensor = test_tensor.to(model._device)
        corpus_tensor = corpus_tensor.to(model._device)

        with torch.no_grad():
            # project and normalize both corpus and test vectors
            proj_corpus = torch.nn.functional.normalize(model(corpus_tensor), p=2, dim=1)
            proj_queries = torch.nn.functional.normalize(model(test_tensor), p=2, dim=1)

            # compute cosine similarity matrix between queries and corpus
            sim_matrix = proj_queries @ proj_corpus.T
            
            # retrieve top 10 to safely drop the anchor itself
            _, top_k = torch.topk(sim_matrix, k=10, dim=1, largest=True)

            hits = 0
            evaluable = 0

            for r_idx, q_idx in enumerate(test_ids):
                q_paper_id = dataset.get_paper_id(q_idx)
                
                # get ground-truth positive ids for the test anchor
                real_pos = dataset.get_positives(q_paper_id)
                
                # if anchor is lack of positives - skip query
                if not real_pos:
                    continue

                evaluable += 1
                
                # filter out anchor itself and grab top 5 candidates
                pred_ids = [all_ids[c] for c in top_k[r_idx] if all_ids[c] != q_paper_id][:5]
                
                # check if any real positive is in predicted top 5
                for pos in real_pos:
                    if pos in pred_ids:
                        hits += 1
                        break

            # calculate recall@5 for current model
            test_recall = hits / evaluable if evaluable > 0 else 0.0
            print(f"Model (Dim {entry['dim']}, Margin {entry['margin']}) -> Test Recall@5: {test_recall:.4f}")

            # track the best performing model
            if test_recall > best_test_recall:
                best_test_recall = test_recall
                winning_model_path = entry["path"]

    # rename the ultimate winner to final weights.pt file
    os.replace(winning_model_path, "./src/model/data/weights.pt")
    
    # clean sweep garbage - delete remaining models from the hard drive
    for entry in GLOBAL_TOP_MODELS:
        if os.path.exists(entry["path"]):
            os.remove(entry["path"])

    print(f"\nFinal Winner Saved to ./src/model/data/weights.pt (Recall@5: {best_test_recall:.4f})")