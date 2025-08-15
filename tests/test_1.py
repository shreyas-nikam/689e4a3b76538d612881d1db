import pytest
from definition_e591f420d9d7415495e080a9f0478e11 import clean_text

@pytest.mark.parametrize("input_text, expected_output", [
    ("  Hello World!  ", "hello world!"),
    ("This is a test.", "this is a test."),
    ("Mixed Case Text", "mixed case text"),
    ("Punctuation!@#$%^&*()_+=-`~[]\{}|;\':\",./<>?", "punctuation!@#$%^&*()_+=-`~[]\{}|;\':\",./<>?"),
    ("Line\nBreaks\rAnd\tTabs", "line\nbreaks\rand\ttabs"),
])
def test_clean_text(input_text, expected_output):
    assert clean_text(input_text) == expected_output
