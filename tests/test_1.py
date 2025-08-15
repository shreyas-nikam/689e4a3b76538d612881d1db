import pytest
from definition_d36c34ec45844f2caa1890586e42b557 import clean_text

@pytest.mark.parametrize("input_text, expected_text", [
    ("  Hello World!  ", "hello world!"),
    ("This is a test.", "this is a test."),
    ("Multiple   spaces   here.", "multiple spaces here."),
    ("Punctuation!?", "punctuation!?"),
    ("", ""),
])
def test_clean_text(input_text, expected_text):
    assert clean_text(input_text) == expected_text
