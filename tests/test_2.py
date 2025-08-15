import pytest
from definition_15ce9f59fbe8459bb05317bfda3c68a1 import load_sentence_bert_model
from sentence_transformers import SentenceTransformer

def test_load_sentence_bert_model_valid_model():
    """Test loading a valid Sentence-BERT model."""
    model = load_sentence_bert_model("all-MiniLM-L6-v2")
    assert isinstance(model, SentenceTransformer)

def test_load_sentence_bert_model_invalid_model():
    """Test loading an invalid/non-existent Sentence-BERT model."""
    with pytest.raises(OSError):
        load_sentence_bert_model("invalid-model-name")

def test_load_sentence_bert_model_none_model_name():
     """Test loading a Sentence-BERT model with None as name."""
     with pytest.raises(TypeError):
        load_sentence_bert_model(None)

def test_load_sentence_bert_model_empty_model_name():
    """Test loading a Sentence-BERT model with an empty string as name."""
    with pytest.raises(OSError):
        load_sentence_bert_model("")

def test_load_sentence_bert_model_numerical_model_name():
    """Test loading a Sentence-BERT model with a numerical name."""
    with pytest.raises(TypeError):
        load_sentence_bert_model(123)