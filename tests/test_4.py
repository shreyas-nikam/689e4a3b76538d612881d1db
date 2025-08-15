import pytest
import numpy as np
from definition_3b2ffda4b12e4fb4b71bd399e9a429f5 import compute_cosine_similarity

@pytest.mark.parametrize("embedding1, embedding2, expected", [
    (np.array([1, 0, 0]), np.array([1, 0, 0]), 1.0),  # Identical vectors
    (np.array([1, 0, 0]), np.array([0, 1, 0]), 0.0),  # Orthogonal vectors
    (np.array([1, 1, 0]), np.array([1, 0, 0]), np.sqrt(2)/2),  # Non-orthogonal vectors
    (np.array([1, 1, 1]), np.array([1, 1, 1]), 1.0),  # Identical vectors again
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.9746318461970762)
])
def test_compute_cosine_similarity(embedding1, embedding2, expected):
    assert np.isclose(compute_cosine_similarity(embedding1, embedding2), expected)
