import pandas as pd
from pypdf import PdfReader

def extract_text_from_pdfs(paths):
    """Extracts text from PDF documents and stores them in a DataFrame."""
    data = []
    for path in paths:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            data.append({'document_id': path, 'title': None, 'text': text})
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
    df = pd.DataFrame(data)
    return df

def clean_text(text):
                """Cleans text by lowercasing and removing extra spaces."""
                text = text.lower()
                text = " ".join(text.split())
                return text

from sentence_transformers import SentenceTransformer

def load_sentence_bert_model(model_name):
    """Loads a Sentence-BERT model."""
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    model = SentenceTransformer(model_name)
    return model

import pandas as pd
import numpy as np

def generate_embeddings(df, model):
    """Generates dense vector embeddings for each document in the DataFrame."""
    if 'text' not in df.columns:
        raise KeyError("The DataFrame must contain a 'text' column.")
    df['embeddings'] = df['text'].apply(lambda x: model.encode(str(x)) if x is not None else model.encode(""))
    return df

import pandas as pd
from atlas_embed import AtlasEmbed

def compute_text_projection(df, text, x, y, neighbors):
    """Computes a 2D projection of text embeddings for visualization."""

    if df.empty:
        raise Exception("DataFrame cannot be empty.")

    if text not in df.columns:
        raise KeyError(f"Text column '{text}' not found in DataFrame.")

    try:
        atlas = AtlasEmbed(df=df, text=text)
        df = atlas.embed(x=x, y=y, neighbors=neighbors)
        return df
    except Exception as e:
        raise Exception(f"Error during embedding: {e}")

import pandas as pd
from sklearn.cluster import KMeans


def apply_kmeans_clustering(df, n_clusters, x_col, y_col):
    """Applies K-Means clustering to the projection coordinates."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df['cluster_id'] = kmeans.fit_predict(df[[x_col, y_col]])
    return df

import numpy as np

def compute_cosine_similarity(embedding1, embedding2):
    """Computes cosine similarity between two embeddings."""
    norm_1 = np.linalg.norm(embedding1)
    norm_2 = np.linalg.norm(embedding2)

    if norm_1 == 0 or norm_2 == 0:
        return np.nan

    similarity = np.dot(embedding1, embedding2) / (norm_1 * norm_2)
    return similarity

import numpy as np
import pandas as pd

def semantic_similarity_search(df, query_embedding, n):
    """Performs semantic similarity search to find the top-N most similar documents to a given query.
    Args:
        df (pd.DataFrame): DataFrame containing document embeddings.
        query_embedding (np.ndarray): Embedding of the query.
        n (int): Number of top matches to return.
    Returns:
        list: List of tuples, where each tuple contains the document ID and its similarity score to the query.
    """
    if n < 1:
        raise ValueError("n must be greater than 0.")
    if df.empty:
        return []

    similarities = []
    for index, row in df.iterrows():
        doc_embedding = row['embeddings']
        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        similarities.append((row['document_id'], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:min(n, len(df))]

def calculate_context_relevancy(query, context):
                """Calculates the Context Relevancy metric."""
                if not query or not context:
                    return 0.0

                query_words = query.lower().split()
                context_words = context.lower().split()

                if not query_words or not context_words:
                    return 0.0

                common_words = set(query_words) & set(context_words)

                relevancy = len(common_words) / len(query_words)

                return relevancy

def calculate_groundedness(answer, context):
                """Calculates the Groundedness metric."""

                if not answer or not context:
                    return 0.0

                answer_words = answer.lower().split()
                context_words = context.lower().split()

                if not answer_words or not context_words:
                    return 0.0

                common_words = set(answer_words) & set(context_words)

                if not common_words:
                    return 0.0

                return len(common_words) / len(answer_words)

def calculate_completeness(context, answer):
                """Calculates the Completeness metric."""
                if not context and not answer:
                    return 1.0
                if not context:
                    return 0.0
                if not answer:
                    return 0.0

                context_words = context.lower().split()
                answer_words = answer.lower().split()

                if not context_words and not answer_words:
                    return 1.0

                if not answer_words:
                    return 0.0

                common_words = set(context_words) & set(answer_words)
                completeness = len(common_words) / len(answer_words) if answer_words else 0.0
                return completeness

def calculate_answer_relevancy(answer, query):
                """Calculates the Answer Relevancy metric."""

                if not answer or not query:
                    return 0.0

                answer_words = answer.lower().split()
                query_words = query.lower().split()

                common_words = set(answer_words) & set(query_words)

                relevancy = len(common_words) / len(query_words) if len(query_words) > 0 else 0.0

                return min(1.0, relevancy)

import pandas as pd
import duckdb

def load_data_from_duckdb(df, predicate):
    """Loads data from DuckDB based on predicate.
    Args:
        df (pd.DataFrame): Input DataFrame.
        predicate (str): DuckDB predicate.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if predicate is None:
        raise TypeError("Predicate cannot be None")

    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        con.register('df', df)
        selection = con.execute(f"SELECT * FROM df WHERE {predicate}").fetchdf()
        con.close()
        return selection
    except Exception as e:
        raise NameError(f"Error executing predicate: {e}")