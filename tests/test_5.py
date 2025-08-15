import pytest
import pandas as pd
from sentence_transformers import SentenceTransformer
from definition_c3b7a49c00694a6b9bb36f0aaa0c25cc import find_similar_documents


@pytest.fixture
def sample_dataframe():
    data = {'document_id': [1, 2, 3],
            'text': ["This is the first document.", "This is the second document.", "This is the third document."]}
    return pd.DataFrame(data)

@pytest.fixture
def sample_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def test_find_similar_documents_with_document_id(sample_dataframe, sample_model):
    result = find_similar_documents(sample_dataframe, 1, sample_model, 2)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 2
    assert 'similarity' in result.columns


def test_find_similar_documents_with_query_string(sample_dataframe, sample_model):
    result = find_similar_documents(sample_dataframe, "first document", sample_model, 2)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 2
    assert 'similarity' in result.columns

def test_find_similar_documents_empty_dataframe(sample_model):
    df = pd.DataFrame()
    result = find_similar_documents(df, "test", sample_model, 1)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_find_similar_documents_invalid_n(sample_dataframe, sample_model):
    with pytest.raises(ValueError):
        find_similar_documents(sample_dataframe, "test", sample_model, -1)
