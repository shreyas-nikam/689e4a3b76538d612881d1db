import pytest
import numpy as np
from unittest.mock import Mock
from definition_c03d40c4b2e448d1808231d69b5ffce4 import generate_embeddings

@pytest.fixture
def mock_model():
    model = Mock()
    model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Mock embeddings
    return model

def test_generate_embeddings_empty_list(mock_model):
    result = generate_embeddings([], mock_model)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 3)

def test_generate_embeddings_valid_input(mock_model):
    texts = ["This is the first text.", "This is the second text."]
    result = generate_embeddings(texts, mock_model)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    assert np.allclose(result, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])) # Checking embedding value

def test_generate_embeddings_single_text(mock_model):
    texts = ["Only one text here."]
    mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])
    result = generate_embeddings(texts, mock_model)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    assert np.allclose(result, np.array([[0.7, 0.8, 0.9]]))

def test_generate_embeddings_model_called(mock_model):
    texts = ["Text 1", "Text 2"]
    generate_embeddings(texts, mock_model)
    mock_model.encode.assert_called_once_with(texts)

def test_generate_embeddings_different_embedding_size(mock_model):
    texts = ["Text 1"]
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    result = generate_embeddings(texts, mock_model)
    assert result.shape == (1, 4)
