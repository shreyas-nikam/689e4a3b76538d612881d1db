import pytest
import numpy as np
from definition_c736a22b12a947eca0f94a3a3c7a0cdc import compute_cosine_similarity

@pytest.mark.parametrize("vector_a, vector_b, expected", [
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.9746318461970762),
    (np.array([1, 0, 0]), np.array([0, 1, 0]), 0.0),
    (np.array([1, 1, 1]), np.array([1, 1, 1]), 1.0),
    (np.array([1, 2]), np.array([1, 2, 3]), ValueError),
    (np.array([0, 0, 0]), np.array([1, 1, 1]), ZeroDivisionError),
])
def test_compute_cosine_similarity(vector_a, vector_b, expected):
    try:
        assert np.isclose(compute_cosine_similarity(vector_a, vector_b), expected)
    except Exception as e:
        assert type(e) == expected
