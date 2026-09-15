import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import wandb

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


PATH_TO_MODEL_WEIGHTS = "./src/model/data/weights.pt"


def train():
    wandb.init()
    config = wandb.config

    connection.init_pool()

    with connection.pool.connection() as conn:
        embeddings: dict[str, list[float]] = fetch_papers_embeddings(conn=conn)
        distance_matrix: dict[str, dict[str, float]] = build_distance_matrix(conn=conn)
        idf_scores: dict[str, float] = fetch_idf_scores(conn=conn)
        keywords: list[list[str]] = fetch_paper_keywords(conn=conn)

    avg_idf_score = sum(idf_scores.values()) / len(idf_scores)

    dataset = PaperPairDataset(
        matrix_distances=distance_matrix,
        embeddings=embeddings,
        idf_scores=idf_scores,
        avg_idf_score=avg_idf_score,
        keywords=keywords,
        margin=config.margin,
    )

    train_size = int(0.9 * len(dataset))
    training_dataset = Subset(dataset=dataset, indices=range(0, train_size))
    validation_dataset = Subset(dataset=dataset, indices=range(train_size, len(dataset)))

    training_dataloader = DataLoader(
        dataset=training_dataset,
        shuffle=True,
        batch_size=32,
        collate_fn=custom_collate_fn,
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        batch_size=16,
        collate_fn=custom_collate_fn,
    )

    # map all indices to real papers' ids and get embedding for every paper
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

    best_recall = 0.0
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
            # get top 6 since self will be always with the highest score
            _, top_6_indices = torch.topk(similarity_matrix, k=6, dim=1, largest=True)
            top_5_corpus_indices = top_6_indices[:, 1:]     # drop self

            hits = 0
            evaluable_queries = 0       # num of queries that have positive vectors

            for row_idx, val_idx in enumerate(val_ids):
                anchor_id = dataset.get_paper_id(val_idx)

                # get all positives based on TF-IDF score from dataset for curr anchor
                real_positives = dataset.get_positives(anchor_id)
                if not real_positives:
                    continue

                evaluable_queries += 1

                predicted_ids = [all_ids[c_idx] for c_idx in top_5_corpus_indices[row_idx]]

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
        if recall_at_5 > best_recall:
            best_recall = recall_at_5
            patience = 0
            model.save(path=PATH_TO_MODEL_WEIGHTS)
            wandb.log({"best_recall": best_recall})
        else:
            patience += 1
            if patience >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

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