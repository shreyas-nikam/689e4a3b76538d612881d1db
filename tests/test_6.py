import pytest
from definition_00ea2e798147445d9663e95d26939f3e import calculate_context_relevancy
from sentence_transformers import SentenceTransformer

@pytest.fixture(scope="module")
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def test_calculate_context_relevancy_valid(model):
    query = "What is the revenue?"
    context = "The revenue is $100."
    relevancy = calculate_context_relevancy(query, context, model)
    assert 0 <= relevancy <= 1

def test_calculate_context_relevancy_empty_query(model):
    query = ""
    context = "The revenue is $100."
    relevancy = calculate_context_relevancy(query, context, model)
    assert relevancy == 0.0

def test_calculate_context_relevancy_empty_context(model):
    query = "What is the revenue?"
    context = ""
    relevancy = calculate_context_relevancy(query, context, model)
    assert relevancy == 0.0

def test_calculate_context_relevancy_no_relevance(model):
    query = "What is the capital of France?"
    context = "The revenue is $100."
    relevancy = calculate_context_relevancy(query, context, model)
    assert 0 <= relevancy <= 1 # Score should be low but not necessarily 0 due to unrelated embeddings

def test_calculate_context_relevancy_identical(model):
    query = "The revenue is $100."
    context = "The revenue is $100."
    relevancy = calculate_context_relevancy(query, context, model)
    assert 0 <= relevancy <= 1
