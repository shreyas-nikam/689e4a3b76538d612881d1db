import pytest
from definition_5d5c0be45ba249bab1e7620184bc61b3 import calculate_context_relevancy

@pytest.mark.parametrize("query, context, expected", [
    ("What is the capital of France?", "Paris is the capital of France.", 1.0),
    ("Meaning of life", "42", 0.0),
    ("Financial report", "This is a financial document.", 0.0),
    ("", "Some context.", 0.0),
    ("Query string", "", 0.0)
])
def test_calculate_context_relevancy(query, context, expected):
    assert calculate_context_relevancy(query, context) == expected