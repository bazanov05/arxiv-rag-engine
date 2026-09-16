import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class PaperEncoder(nn.Module):
    """
    A fine-tunable paper embedding model with a learned projection head.

    Wraps a pretrained scientific SentenceTransformer generating 768-dimensional embeddings,
    then projects them to a specialized 128-dimensional space trained with triplet loss.
    The projection head discards general text structure and retains the semantic
    dimensions aligned with citation and keyword similarity.

    During training, forward() receives pre-computed 768-dim vectors from the DataLoader
    and passes them through the projection head only. After training, generate_embeddings()
    runs the full pipeline from raw text to normalized 128-dim vectors for storage in pgvector.

    Args:
        model_name: HuggingFace model identifier for the base transformer.
                    Defaults to sentence-transformers/allenai-specter.
        projection_dim: Target output dimension for the learned representation. Defaults to 128.
    """
    def __init__(self, model_name: str = "sentence-transformers/allenai-specter", projection_dim: int = 128):
        super().__init__()

        self._device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name_or_path=model_name, device=self._device)
        self._projection = nn.Linear(in_features=768, out_features=projection_dim)
        self._dropout = nn.Dropout(p=0.25)
        self.to(device=self._device)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Projects pre-computed embeddings to the specialized semantic similarity space.

        Expects 768-dim vectors from the DataLoader. Applies dropout, linear projection,
        and L2-normalizes the output so cosine similarity equals the dot product.

        Args:
            embeddings: Batch of 768-dim embedding tensors of shape (batch_size, 768).

        Returns:
            L2-normalized tensors of shape (batch_size, projection_dim).
        """
        embeddings = self._dropout(embeddings)
        embeddings = self._projection(embeddings)
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates projection-head embeddings from raw title and abstract strings.

        Encodes text with the pretrained base transformer, projects to the target dimension,
        and L2-normalizes the result under torch.no_grad().

        Args:
            texts: List of raw paper text strings to embed.

        Returns:
            List of L2-normalized float vectors, one per input text.
        """
        with torch.no_grad():
            vectors_before_projection = self._model.encode(texts, batch_size=32, show_progress_bar=True)
            vectors_before_projection = torch.tensor(data=vectors_before_projection, device=self._device)

            vectors_after_projection = self._projection(vectors_before_projection)
            vectors_after_projection = nn.functional.normalize(vectors_after_projection, p=2, dim=1)

        return vectors_after_projection.tolist()

    def save(self, path: str) -> None:
        """
        Saves the projection head weights to disk.

        Args:
            path: File path where the checkpoint will be saved, e.g. './models/weights.pt'.
        """
        torch.save({
            "projection_state_dict": self._projection.state_dict(),
        }, path)
