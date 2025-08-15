import pytest
from definition_16248047e1034268b7455635e25467f1 import calculate_context_relevancy
import sentence_transformers

@pytest.fixture(scope="module")
def model():
    return sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def test_calculate_context_relevancy_basic(model):
    query = "What is the revenue?"
    context = "The revenue is $1 million."
    score = calculate_context_relevancy(query, context, model)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_calculate_context_relevancy_no_relevance(model):
    query = "What is the capital of France?"
    context = "The revenue is $1 million."
    score = calculate_context_relevancy(query, context, model)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
def test_calculate_context_relevancy_empty_query(model):
    query = ""
    context = "The revenue is $1 million."
    score = calculate_context_relevancy(query, context, model)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_calculate_context_relevancy_empty_context(model):
    query = "What is the revenue?"
    context = ""
    score = calculate_context_relevancy(query, context, model)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_calculate_context_relevancy_complex_sentences(model):
    query = "What were the factors contributing to the company's success, and how does the future look?"
    context = "Our success was mainly influenced by market demand, innovative products, and cost-effective operations. The future appears promising with new expansion plans and a strong customer base."
    score = calculate_context_relevancy(query, context, model)
    assert isinstance(score, float)
    assert 0 <= score <= 1
