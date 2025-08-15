import pytest
from definition_e26805cadf2341dba889296f637bfcdb import clean_text

@pytest.mark.parametrize("input, expected", [
    ("  Hello World!  ", "hello world!"),
    ("MixedCase Text.", "mixedcase text."),
    ("No extra   spaces here. ", "no extra spaces here."),
    ("Punctuation, marks!?", "punctuation, marks!?"),
    ("", "")
])

def test_clean_text(input, expected):
    assert clean_text(input) == expected
