import pytest
from definition_bd626032a2b34f60959cc682d7ee73c2 import split_text_into_sentences

@pytest.mark.parametrize("input_text, expected_output", [
    ("This is sentence one. This is sentence two.", ["This is sentence one.", "This is sentence two."]),
    ("", []),
    ("Sentence with no ending", ["Sentence with no ending"]),
    ("One. Two. Three!", ["One.", "Two.", "Three!"]),
    ("Sentence with Mr. Jones.", ["Sentence with Mr. Jones."])
])
def test_split_text_into_sentences(input_text, expected_output):
    assert split_text_into_sentences(input_text) == expected_output
