import pytest
from definition_459da685e6614000b30182bdbc1eebf8 import split_text_into_sentences

@pytest.mark.parametrize("input_text, expected_sentences", [
    ("This is a sentence. This is another sentence.", ["This is a sentence.", "This is another sentence."]),
    ("Only one sentence.", ["Only one sentence."]),
    ("", []),
    ("Sentence with a Mr. and Mrs. in it.", ["Sentence with a Mr. and Mrs. in it."]),
    ("Sentence with a number 123.", ["Sentence with a number 123."])
])
def test_split_text_into_sentences(input_text, expected_sentences):
    assert split_text_into_sentences(input_text) == expected_sentences
