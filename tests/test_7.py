import pytest
from definition_1da6e4e226624502b2a2711df2925b7f import calculate_groundedness
from sentence_transformers import SentenceTransformer

@pytest.fixture
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def mock_sentence_transformer(sentences):
    # Mock embedding model - replace with appropriate mocking strategy if needed
    return [([float(i) for i in range(384)]) for _ in sentences]

def test_calculate_groundedness_high_groundedness(model, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_sentence_transformer)
    answer = "The company's revenue grew by 10% in Q3."
    context = "The company reported a 10% increase in Q3 revenue."
    groundedness = calculate_groundedness(answer, context, model)
    assert groundedness > 0.8

def test_calculate_groundedness_low_groundedness(model, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_sentence_transformer)
    answer = "The company's revenue grew by 20% in Q3."
    context = "The company reported a 10% increase in Q3 revenue."
    groundedness = calculate_groundedness(answer, context, model)
    assert groundedness < 0.5

def test_calculate_groundedness_empty_answer(model, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_sentence_transformer)
    answer = ""
    context = "The company reported a 10% increase in Q3 revenue."
    groundedness = calculate_groundedness(answer, context, model)
    assert groundedness == 0.0

def test_calculate_groundedness_empty_context(model, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_sentence_transformer)
    answer = "The company's revenue grew by 10% in Q3."
    context = ""
    groundedness = calculate_groundedness(answer, context, model)
    assert groundedness == 0.0

def test_calculate_groundedness_identical_answer_context(model, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_sentence_transformer)
    answer = "The company reported a 10% increase in Q3 revenue."
    context = "The company reported a 10% increase in Q3 revenue."
    groundedness = calculate_groundedness(answer, context, model)
    assert groundedness == 1.0