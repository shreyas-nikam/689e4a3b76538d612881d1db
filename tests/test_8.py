import pytest
from definition_8aeaaf232bde4112beb18a8eb85988e6 import calculate_completeness
from sentence_transformers import SentenceTransformer

@pytest.fixture
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def test_completeness_full_coverage(model):
    context = "The company's revenue increased. Profits also increased."
    answer = "Revenue and profits increased for the company."
    completeness = calculate_completeness(context, answer, model)
    assert completeness > 0.8

def test_completeness_partial_coverage(model):
    context = "The company's revenue increased. Profits also increased."
    answer = "Revenue increased for the company."
    completeness = calculate_completeness(context, answer, model)
    assert 0.4 < completeness < 0.8

def test_completeness_no_coverage(model):
    context = "The company's revenue increased. Profits also increased."
    answer = "The weather is nice today."
    completeness = calculate_completeness(context, answer, model)
    assert completeness < 0.2

def test_completeness_empty_context(model):
    context = ""
    answer = "Revenue increased for the company."
    completeness = calculate_completeness(context, answer, model)
    assert completeness == 0.0

def test_completeness_empty_answer(model):
    context = "The company's revenue increased. Profits also increased."
    answer = ""
    completeness = calculate_completeness(context, answer, model)
    assert completeness < 0.2

