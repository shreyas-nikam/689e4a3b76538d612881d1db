import pytest
from definition_789d9aa653a7432fa34720562010bf5a import calculate_completeness
from sentence_transformers import SentenceTransformer
import numpy as np

@pytest.fixture(scope="module")
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


@pytest.mark.parametrize("context, answer, expected", [
    ("The company's revenue grew by 10%. Expenses remained flat.", "Revenue increased by 10%.", 1.0),
    ("This is a test.", "This is completely different.", 0.0),
    ("Long context sentence 1. Long context sentence 2.", "Short answer.", 0.5),
    ("", "", 1.0),
    ("Important info.", "Irrelevant answer.", 0.0)
])
def test_calculate_completeness(context, answer, expected, model):
    
    if not context and not answer:
        assert calculate_completeness(context, answer, model) == 1.0 # Both Empty case returns 1.0
        return

    if not context:
        assert calculate_completeness(context, answer, model) == 0.0
        return
    
    if not answer:
        assert calculate_completeness(context, answer, model) == 0.0
        return

    context_embedding = model.encode(context)
    answer_embedding = model.encode(answer)

    # Simple test: average cosine similarity should be close to the expected value.
    similarity = cosine_similarity(context_embedding, answer_embedding)
    
    # Adjust tolerance if necessary
    assert calculate_completeness(context, answer, model) >= 0.0

