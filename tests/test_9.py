import pytest
from definition_239d22546d544f20bee3213d99c373bd import calculate_groundedness

@pytest.mark.parametrize("answer, context, expected", [
    ("The sky is blue.", "The sky is blue and the grass is green.", 1.0),
    ("Paris is the capital of France.", "France is a country in Europe.", 0.0),
    ("The cat sat on the mat.", "The dog barked loudly.", 0.0),
    ("", "Some context.", 0.0),
    ("An answer.", "", 0.0),
])
def test_calculate_groundedness(answer, context, expected):
    assert calculate_groundedness(answer, context) == expected
