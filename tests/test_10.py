import pytest
from definition_3885ce82bcdb443494c05793dcbb2c77 import calculate_completeness

@pytest.mark.parametrize("context, answer, expected", [
    ("The sky is blue.", "The sky is blue.", 1.0),
    ("The sky is blue.", "The sky is not blue.", 0.0),
    ("The cat sat on the mat.", "cat mat", 1.0),
    ("This is a long context about financial documents and their similarity.", "financial documents", pytest.approx(0.5, abs=0.1)),
    ("", "", 1.0), 
])
def test_calculate_completeness(context, answer, expected):
    assert calculate_completeness(context, answer) == expected
