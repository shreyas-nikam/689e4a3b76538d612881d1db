import pytest
import pandas as pd
from sentence_transformers import SentenceTransformer
from definition_63766d46f16a48659e5209eeeffa1c5e import find_similar_documents

@pytest.fixture
def sample_dataframe():
    data = {'document_id': [1, 2, 3],
            'text': ["This is document 1 about finance.", "Document 2 discusses investment strategies.", "Document 3 covers market analysis."],
            'embeddings': [None, None, None]}
    df = pd.DataFrame(data)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['embeddings'] = df['text'].apply(lambda x: model.encode(x))
    return df, model

def test_find_similar_documents_with_document_id(sample_dataframe):
    df, model = sample_dataframe
    N = 2
    query = 1
    result_df = find_similar_documents(df, query, model, N)
    assert len(result_df) == N
    assert 'similarity_score' in result_df.columns
    assert result_df['document_id'].iloc[0] == 1 #Most similar should be the document itself

def test_find_similar_documents_with_text_query(sample_dataframe):
    df, model = sample_dataframe
    N = 2
    query = "investment analysis"
    result_df = find_similar_documents(df, query, model, N)
    assert len(result_df) == N
    assert 'similarity_score' in result_df.columns
    assert result_df['document_id'].iloc[0] == 2 #Doc 2 should be most similar
    assert result_df['document_id'].iloc[1] == 3 #Doc 3 should be next similar

def test_find_similar_documents_empty_dataframe():
    df = pd.DataFrame({'document_id': [], 'text': [], 'embeddings': []})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    N = 2
    query = "finance"
    result_df = find_similar_documents(df, query, model, N)
    assert len(result_df) == 0

def test_find_similar_documents_invalid_query_id(sample_dataframe):
    df, model = sample_dataframe
    N = 2
    query = 4  # ID not in the dataframe
    with pytest.raises(KeyError):
        find_similar_documents(df, query, model, N)

def test_find_similar_documents_n_greater_than_dataframe_size(sample_dataframe):
    df, model = sample_dataframe
    N = 5  # Greater than the number of documents
    query = "finance"
    result_df = find_similar_documents(df, query, model, N)
    assert len(result_df) == len(df)
