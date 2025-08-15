import pytest
import numpy as np
from definition_49ef4ca7fb194e02bded9bfab77c1a00 import compute_cosine_similarity

@pytest.mark.parametrize("embedding1, embedding2, expected", [
    (np.array([1, 0, 0]), np.array([1, 0, 0]), 1.0),  # Identical vectors
    (np.array([1, 0, 0]), np.array([0, 1, 0]), 0.0),  # Orthogonal vectors
    (np.array([1, 1, 0]), np.array([1, 0, 0]), 0.70710678),  # Vectors with some similarity
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.97463184),  # General case
    (np.array([0, 0, 0]), np.array([0, 0, 0]), np.nan),  # Zero vectors, expect NaN
])
def test_compute_cosine_similarity(embedding1, embedding2, expected):
    if np.isnan(expected):
        assert np.isnan(compute_cosine_similarity(embedding1, embedding2))
    else:
        assert np.isclose(compute_cosine_similarity(embedding1, embedding2), expected, rtol=1e-6)
