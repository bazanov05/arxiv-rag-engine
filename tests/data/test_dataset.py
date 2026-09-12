import pytest
import numpy as np
import torch
from src.data.dataset import PaperPairDataset, custom_collate_fn


@pytest.fixture
def mock_dataset_data():
    """
    Provides a controlled dataset to test all triplet mining conditions.
    
    Setup details:
    - avg_idf_score = 2.0 -> Positive threshold = 2 * 2.0 = 4.0
    - idf_scores: 'transformer': 3.0, 'attention': 2.5, 'quantum': 4.0, 'graph': 1.5
    
    Paper 1 is our anchor (index 0):
    - Paper 1 & Paper 2: share 'transformer' (3.0) + 'attention' (2.5). Score = 5.5 >= 4.0 (Positive)
    - Paper 1 & Paper 3: share 'attention' (2.5). Score = 2.5 < 4.0 (Negative)
    - Paper 1 & Paper 4: share no keywords. Score = 0.0 < 4.0 (Negative)
    """
    embeddings = {
        "2301.0001": [1.0],
        "2301.0002": [2.0],
        "2301.0003": [3.0],
        "2301.0004": [4.0]
    }

    keywords = [
        ["transformer", "attention", "quantum"],  # 2301.0001
        ["transformer", "attention"],             # 2301.0002
        ["attention"],                            # 2301.0003
        ["graph"]                                 # 2301.0004
    ]

    idf_scores = {
        "transformer": 3.0,
        "attention": 2.5,
        "quantum": 4.0,
        "graph": 1.5
    }

    # Default uniform distances to be overwritten in specific test cases
    matrix_distances = {
        "2301.0001": {"2301.0002": 0.5, "2301.0003": 0.5, "2301.0004": 0.5},
        "2301.0002": {"2301.0001": 0.5, "2301.0003": 0.5, "2301.0004": 0.5},
        "2301.0003": {"2301.0001": 0.5, "2301.0002": 0.5, "2301.0004": 0.5},
        "2301.0004": {"2301.0001": 0.5, "2301.0002": 0.5, "2301.0003": 0.5},
    }

    return {
        "matrix_distances": matrix_distances,
        "embeddings": embeddings,
        "idf_scores": idf_scores,
        "avg_idf_score": 2.0,
        "keywords": keywords,
        "margin": 0.2
    }


def test_dataset_len(mock_dataset_data):
    """Tests if the dataset returns the correct total number of papers."""
    dataset = PaperPairDataset(**mock_dataset_data)
    assert len(dataset) == 4


def test_precomputed_candidates(mock_dataset_data):
    """Tests if IDF keyword similarity correctly partitions papers into positives and negatives."""
    dataset = PaperPairDataset(**mock_dataset_data)

    positives = dataset._positives["2301.0001"]
    negatives = dataset._negatives["2301.0001"]

    # Paper 2 has score 5.5 >= 4.0 -> Positive
    # Paper 3 has score 2.5 < 4.0 -> Negative
    # Paper 4 has score 0.0 < 4.0 -> Negative
    assert positives == ["2301.0002"]
    assert sorted(negatives) == ["2301.0003", "2301.0004"]


def test_getitem_semi_hard_logic(mock_dataset_data):
    """Tests the optimal tier: negative is further than positive, but within margin."""
    # Setup Semi-Hard: d_pos < d_neg < d_pos + margin (0.3 < 0.4 < 0.5)
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0002"] = 0.3  # Positive
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0003"] = 0.4  # Semi-hard Negative
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0004"] = 0.9  # Easy Negative

    dataset = PaperPairDataset(**mock_dataset_data)

    # Index 0 corresponds to "2301.0001"
    anchor, pos, neg = dataset[0]

    assert anchor == [1.0]
    assert pos == [2.0]
    assert neg == [3.0]


def test_getitem_hard_fallback(mock_dataset_data):
    """Tests fallback to a hard triplet (d_neg < d_pos) when no semi-hard candidate exists."""
    # Setup Hard: d_neg < d_pos
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0002"] = 0.6  # Positive
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0003"] = 0.4  # Hard Negative
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0004"] = 0.9  # Easy Negative

    dataset = PaperPairDataset(**mock_dataset_data)
    anchor, pos, neg = dataset[0]

    assert anchor == [1.0]
    assert pos == [2.0]
    assert neg == [3.0]


def test_getitem_easy_random_fallback(mock_dataset_data):
    """Tests fallback to random sampling when neither semi-hard nor hard candidates exist."""
    # Setup Easy: d_neg > d_pos + margin (0.1 + 0.2 = 0.3 < 0.8 and 0.9)
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0002"] = 0.1  # Positive
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0003"] = 0.8  # Easy Negative
    mock_dataset_data["matrix_distances"]["2301.0001"]["2301.0004"] = 0.9  # Easy Negative

    dataset = PaperPairDataset(**mock_dataset_data)

    np.random.seed(42)
    anchor, pos, neg = dataset[0]

    assert anchor == [1.0]
    assert pos == [2.0]
    assert neg in ([3.0], [4.0])


def test_getitem_edge_case_empty_positives(mock_dataset_data):
    """Tests that (anchor, None, None) is returned when a paper has no positive candidates."""
    # Change keywords so paper 1 has 0 overlap with any other paper
    mock_dataset_data["keywords"] = [
        ["unrelated"],
        ["transformer"],
        ["attention"],
        ["graph"]
    ]

    dataset = PaperPairDataset(**mock_dataset_data)
    anchor, pos, neg = dataset[0]

    assert anchor == [1.0]
    assert pos is None
    assert neg is None


def test_custom_collate_fn_filtering():
    """Tests that custom_collate_fn removes invalid triplets and formats tensors."""
    valid_sample = ([1.0] * 768, [2.0] * 768, [3.0] * 768)
    invalid_sample = ([4.0] * 768, None, None)

    anchors, positives, negatives = custom_collate_fn([valid_sample, invalid_sample])

    assert anchors.shape == (1, 768)
    assert positives.shape == (1, 768)
    assert negatives.shape == (1, 768)


def test_custom_collate_fn_all_invalid():
    """Tests that custom_collate_fn returns empty tensors of shape (0, 768) if all samples are invalid."""
    invalid_sample = ([1.0] * 768, None, None)

    anchors, positives, negatives = custom_collate_fn([invalid_sample])

    assert anchors.shape == (0, 768)
    assert positives.shape == (0, 768)
    assert negatives.shape == (0, 768)