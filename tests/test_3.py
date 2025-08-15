import pytest
import pandas as pd
from sentence_transformers import SentenceTransformer
from definition_8b14596f0f3349fb8bae1003b2a33f14 import generate_embeddings
import numpy as np

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'text': ["This is the first document.", "This is the second document."]
    })

@pytest.fixture
def mock_model():
    class MockModel:
        def encode(self, sentences):
            if isinstance(sentences, str):
                return np.array([1.0, 2.0])  # Return a sample embedding for a single sentence
            return [np.array([1.0, 2.0]), np.array([3.0, 4.0])]  # Return dummy embeddings
    return MockModel()

def test_generate_embeddings_success(sample_dataframe, mock_model):
    df = generate_embeddings(sample_dataframe.copy(), mock_model)
    assert 'embeddings' in df.columns
    assert len(df['embeddings']) == 2
    assert isinstance(df['embeddings'][0], np.ndarray)
    assert df['embeddings'][0].shape == (2,)

def test_generate_embeddings_empty_dataframe(mock_model):
    df = pd.DataFrame({'text': []})
    df = generate_embeddings(df.copy(), mock_model)
    assert 'embeddings' in df.columns
    assert len(df['embeddings']) == 0

def test_generate_embeddings_missing_text_column(sample_dataframe, mock_model):
    df = sample_dataframe.copy().drop(columns=['text'])
    with pytest.raises(KeyError):
        generate_embeddings(df, mock_model)

def test_generate_embeddings_none_text(sample_dataframe, mock_model):
    df = pd.DataFrame({'text': [None]})
    df = generate_embeddings(df.copy(), mock_model)
    assert 'embeddings' in df.columns
    assert len(df['embeddings']) == 1
    assert isinstance(df['embeddings'][0], np.ndarray)
    assert df['embeddings'][0].shape == (2,)

def test_generate_embeddings_numerical_text(sample_dataframe, mock_model):
    df = pd.DataFrame({'text': [123, 456]})
    df = generate_embeddings(df.copy(), mock_model)
    assert 'embeddings' in df.columns
    assert len(df['embeddings']) == 2
    assert isinstance(df['embeddings'][0], np.ndarray)
    assert df['embeddings'][0].shape == (2,)
