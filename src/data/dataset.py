import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict



class PaperPairDataset(Dataset):
    """
    A PyTorch Dataset that prepares academic paper representations for contrastive metric learning.

    This class coordinates graph-level paper relationships by combining continuous geometric
    distances from base SPECTER2 embeddings with discrete lexical overlap derived from corpus-level
    TF-IDF statistics. During initialization, it precalculates symmetric keyword overlap
    scores across all papers using set intersections weighted by Inverse Document Frequency (IDF).
    It partitions each paper's candidate pool into positive and negative sets using an empirical
    threshold (2 * average IDF).

    During sample retrieval, the dataset dynamically mines triplets using a three-tier strategy:
    1. Semi-hard triplets: Pairs where the negative candidate is farther from the anchor than the
       positive candidate, but violates the margin constraint (d_pos < d_neg < d_pos + margin).
    2. Hard triplets: Pairs where the negative candidate is strictly closer to the anchor than
       the positive candidate (d_neg < d_pos).
    3. Easy triplets: Randomly sampled positive and negative candidates when no harder alternatives
       satisfy the geometric constraints.
    """
    def __init__(
            self, 
            matrix_distances: dict[str, dict[str, float]], 
            embeddings: dict[str, list[float]],
            idf_scores: dict[str, float],
            avg_idf_score: float,
            keywords: list[list[str]],
            margin: float = 0.2
        ):
        """
        Initializes the dataset, stores metadata, and builds positive and negative candidate pools.

        Args:
            matrix_distances: Nested dictionary representing pairwise geometric distances between paper embeddings,
                              keyed by unique string ArXiv IDs.
            embeddings: Dictionary mapping ArXiv IDs to their corresponding dense feature vectors.
            idf_scores: Dictionary mapping individual vocabulary tokens to their smooth IDF weights.
            avg_idf_score: The mean IDF weight across the vocabulary, used as a baseline scaling unit.
            keywords: Two-dimensional list containing extracted top-K TF-IDF keywords for each paper.
            margin: Margin value used to define boundary violations during semi-hard triplet mining.

        Returns:
            None
        """
        super().__init__()

        self._matrix_distances = matrix_distances
        self._embeddings = embeddings
        self._idf_scores = idf_scores           # dict - word: idf_score
        self._avg_idf_score = avg_idf_score     # avg idf score across all words in all papers
        self._keywords = keywords               # matrix of num_of_papers X top_k_words

        # distance between pos and anchor should be smaller than distance between neg and anchor
        # at least by margin
        self._margin = margin

        # DataLoader will give indices in range [0, N-1], cause we have N papers
        # but paper_ids are not in this range - there are specialized strings
        # so we need to map those indices to real paper_ids
        self._paper_ids = list(embeddings.keys())

        similarity_scores = self._compute_similarity_scores()
        self._positives, self._negatives = self._build_candidate_lists(scores=similarity_scores)

    def _compute_similarity_scores(self) -> dict[str, dict[str, float]]:
        """
        Calculates symmetric pairwise keyword similarity scores for all document combinations in the corpus.

        Converts keyword lists into sets for constant-time lookups and leverages symmetry to compute
        only the upper triangular index pairs before reflecting values across the diagonal.

        Args:
            None

        Returns:
            dict[str, dict[str, float]]: Nested mapping where keys are ArXiv IDs and inner dictionaries
            contain the IDF-weighted intersection score against other papers.
        """
        similarity_scores: dict[str, dict[str, float]] = dict()

        # replace every list with set for fast intersections checks
        for i, keywords_per_paper in enumerate(self._keywords):
            self._keywords[i] = set(keywords_per_paper)

        for i, id1 in enumerate(self._paper_ids):
            # check if id1 has never been processed before
            if id1 not in similarity_scores:
                similarity_scores[id1] = {}
            id1_keywords = self._keywords[i]

            for j in range(i+1, len(self._paper_ids)):
                id2_keywords = self._keywords[j]
                id2 = self._paper_ids[j]

                if id2 not in similarity_scores:
                    similarity_scores[id2] = {}
                
                # get intersection of two sets
                words_in_common = id1_keywords & id2_keywords
                # score is just the same of ids scores of words in common
                score = sum(self._idf_scores[word] for word in words_in_common)

                similarity_scores[id1][id2] = score
                similarity_scores[id2][id1] = score
            
        return similarity_scores

    def _build_candidate_lists(
            self, 
            scores: dict[str, dict[str, float]]
        ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        Partitions candidate papers into positive and negative sets based on an IDF-scaled threshold.

        Separates candidates by evaluating whether the shared keyword score meets or exceeds twice
        the corpus average IDF score (2 * avg_idf_score).

        Args:
            scores: Nested dictionary containing pairwise IDF intersection scores between all papers.

        Returns:
            tuple[dict[str, list[str]], dict[str, list[str]]]: A two-element tuple consisting of:
            - positives: Dictionary mapping each ArXiv ID to candidate IDs meeting the positive threshold.
            - negatives: Dictionary mapping each ArXiv ID to candidate IDs falling below the threshold.
        """
        # build positive and negative candidates
        positives = defaultdict(list)
        negatives = defaultdict(list)

        for id1 in scores.keys():
            for id2, score in scores[id1].items():
                if score >= 1.75 * self._avg_idf_score:
                    positives[id1].append(id2)
                else:
                    negatives[id1].append(id2)

        return positives, negatives

    def _sample_triplet(self, index: int) -> tuple[list[float], list[float] | None, list[float] | None]:
        """
        Extracts a single triplet sample (anchor, positive, negative) according to margin difficulty constraints.

        Iterates over randomly shuffled positive and negative candidate pools. Prioritizes semi-hard
        negatives, falls back to the most violated hard negative pairs, and finally resorts to uniform
        random selection if no constrained boundaries are violated.

        Args:
            index: Integer index referencing the anchor paper inside the internal index map.

        Returns:
            tuple[list[float], list[float] | None, list[float] | None]: A three-element tuple consisting of:
            - anchor: Embedding vector for the target document.
            - positive: Selected positive embedding vector, or None if candidate pools are empty.
            - negative: Selected negative embedding vector, or None if candidate pools are empty.
        """
        paper_id = self._paper_ids[index]
        anchor = self._embeddings[paper_id]

        # get positive and negative candidates for anchor
        positives = self._positives[paper_id]
        negatives = self._negatives[paper_id]

        # if anchor is lack of some candidates - return Nones
        if not positives or not negatives:
            return anchor, None, None

        # shuffle ids
        np.random.shuffle(positives)
        np.random.shuffle(negatives)

        # track best case for hard logic in case semi-hard is not found
        best_hard_pos = None
        best_hard_neg = None

        for positive_id in positives:
            positive = self._embeddings[positive_id]
            for negative_id in negatives:
                negative = self._embeddings[negative_id]

                d_pos = self._matrix_distances[paper_id][positive_id]
                d_neg = self._matrix_distances[paper_id][negative_id]

                # semi-hard logic: d_pos is smaller than d_neg but not by margin
                if d_pos < d_neg and d_neg < d_pos + self._margin:
                    return anchor, positive, negative

                # hard logic: d_pos > d_neg
                if d_neg < d_pos:
                    best_hard_neg = negative
                    best_hard_pos = positive

        # return hard vectors in case semi-hard was not found and hard was found
        if best_hard_pos is not None and best_hard_neg is not None:
            return anchor, best_hard_pos, best_hard_neg

        # otherwise, use easy logic - return random positive and random negative
        random_pos_id = np.random.choice(positives)
        random_neg_id = np.random.choice(negatives)

        positive = self._embeddings[random_pos_id]
        negative = self._embeddings[random_neg_id]

        return anchor, positive, negative

    def __len__(self):
        """
        Reports the total number of paper entities available in the dataset.

        Args:
            None.

        Returns:
            int: Number of items contained in the embeddings collection.
        """
        return len(self._embeddings)

    def __getitem__(self, index) -> tuple[list[float], list[float] | None, list[float] | None]:
        """
        Coordinates batch retrieval by routing integer index requests to the triplet sampling pipeline.

        Args:
            index: Sequential index provided by the PyTorch DataLoader.

        Returns:
            tuple[list[float], list[float] | None, list[float] | None]: Triplet tuple containing the anchor vector,
            selected positive vector (or None), and selected negative vector (or None).
        """
        return self._sample_triplet(index=index)

    def get_paper_id(self, index: int) -> str:
        """Returns the paper_id string corresponding to an integer index."""
        return self._paper_ids[index]

    def get_positives(self, paper_id: str) -> list[str]:
        """Returns the list of ground-truth positive IDs for a given paper_id."""
        return self._positives.get(paper_id, [])
        
    def get_embedding(self, paper_id: str) -> list[float]:
        """Returns the raw base embedding for a given paper_id."""
        return self._embeddings[paper_id]


def custom_collate_fn(triplets: list[tuple[list[float], list[float] | None, list[float] | None]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates raw triplet samples into PyTorch batch tensors while filtering degenerate cases.

    Examines incoming samples to drop any records where mining could not find a valid positive
    or negative embedding candidate. Converts surviving vectors into stacked tensors or returns
    dimensionally consistent empty tensors if no valid samples remain.

    Args:
        triplets: List of raw sample tuples produced by PaperPairDataset.__getitem__, where positive
                  and negative components may contain None values.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A three-element tuple of tensors:
        - anchors: Tensor of shape (batch_size, 768) containing anchor embeddings.
        - positives: Tensor of shape (batch_size, 768) containing positive candidate embeddings.
        - negatives: Tensor of shape (batch_size, 768) containing negative candidate embeddings.
        If no valid triplets exist, returns three empty tensors of shape (0, 768).
    """
    # clean triplet - delete those where positive and negative where not found
    cleaned_triplets = [triplet for triplet in triplets if triplet[1] is not None and triplet[2] is not None]

    # if no triplets are left after the clean - return 3 empty tensors
    if not cleaned_triplets:
        return torch.empty(0, 768), torch.empty(0, 768), torch.empty(0, 768)

    # otherwise return 3 tensors: anchors, positives and negatives 
    anchors = torch.tensor(data=[triplet[0] for triplet in cleaned_triplets])
    positives = torch.tensor(data=[triplet[1] for triplet in cleaned_triplets])
    negatives = torch.tensor(data=[triplet[2] for triplet in cleaned_triplets])

    return anchors, positives, negatives
