import pytest
from definition_696f057b80d9406f997cd6eabacbf3c9 import calculate_groundedness
from sentence_transformers import SentenceTransformer

@pytest.fixture(scope="module")
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@pytest.mark.parametrize("answer, context, expected", [
    ("The company's revenue grew.", "The company reported revenue growth.", 0.9),
    ("The company made a profit.", "No profit was reported.", 0.0),
    ("The cat sat on the mat.", "The dog barked loudly.", 0.0),
])
def test_calculate_groundedness_positive(model, answer, context, expected):
    score = calculate_groundedness(answer, context, model)
    assert score >= 0.0 and score <= 1.0

@pytest.mark.parametrize("answer, context", [
    ("", "The context"),
    ("The answer", ""),
    ("", ""),
])
def test_calculate_groundedness_empty_input(model, answer, context):
    score = calculate_groundedness(answer, context, model)
    assert score >= 0.0 and score <= 1.0

def test_calculate_groundedness_same_sentences(model):
    text = "The company is doing well."
    score = calculate_groundedness(text, text, model)
    assert score >= 0.8

def test_calculate_groundedness_long_text(model):
    answer = "This is a very long answer with many sentences. It should still work." * 3
    context = "This is a very long context with many sentences. It should also work." * 5
    score = calculate_groundedness(answer, context, model)
    assert score >= 0.0 and score <= 1.0

def test_calculate_groundedness_complex_sentences(model):
    answer = "The company's revenue increased, which is a positive sign."
    context = "Revenue increased; this indicates positive growth."
    score = calculate_groundedness(answer, context, model)
    assert score >= 0.6
