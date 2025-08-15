import pypdf
def extract_text_from_pdf(file_path):
    """Extracts text content from a PDF file."""

    if file_path is None:
        raise TypeError("File path cannot be None")

    try:
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")

def clean_text(text):
                """Cleans raw text."""
                text = text.lower()
                text = " ".join(text.split())
                return text

import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(texts, model):
    """Generates text embeddings."""
    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings.")
    embeddings = model.encode(texts)
    return embeddings

import numpy as np
from sklearn.cluster import KMeans

def apply_kmeans_clustering(embeddings, n_clusters):
    """Applies K-Means clustering to embeddings.
    Args:
        embeddings: 2D embeddings (numpy array).
        n_clusters: Number of clusters.
    Returns:
        Cluster labels.
    Raises:
        ValueError: If n_clusters is invalid or embeddings is empty.
    """
    if not isinstance(embeddings, np.ndarray):
        raise AttributeError("Embeddings must be a numpy array.")

    if embeddings.size == 0:
        raise ValueError("Embeddings cannot be empty.")
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be greater than 0.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto')  # Explicitly set n_init
    labels = kmeans.fit_predict(embeddings)
    return labels

import numpy as np

def compute_cosine_similarity(embedding1, embedding2):
    """Computes the cosine similarity between two embedding vectors."""
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)

import pandas as pd
import numpy as np
from sentence_transformers import util

def find_similar_documents(df, query, model, N):
    """Finds and ranks documents based on their cosine similarity to a given query embedding.
    Args:
        df: DataFrame with 'embeddings' column.
        query: Query string or document ID.
        model: SentenceTransformer model.
        N: Number of top similar documents to return.
    Returns:
        DataFrame of the top N similar documents and their scores.
    """
    if df.empty:
        return pd.DataFrame()

    if isinstance(query, int):
        if query not in df['document_id'].values:
            raise KeyError("Query ID not found in dataframe")
        query_embedding = df[df['document_id'] == query]['embeddings'].iloc[0]
    else:
        query_embedding = model.encode(query)

    corpus_embeddings = np.stack(df['embeddings'].to_numpy())
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    df['similarity_score'] = cos_scores.cpu().tolist()

    df_sorted = df.sort_values(by='similarity_score', ascending=False)
    result_df = df_sorted.head(N)

    return result_df[['document_id', 'text', 'similarity_score']].reset_index(drop=True)

from sentence_transformers import util

def calculate_context_relevancy(query, context, model):
    """Calculates context relevancy score between query and context."""
    if not query or not context:
        return 0.0

    query_embedding = model.encode(query, convert_to_tensor=True)
    context_embedding = model.encode(context, convert_to_tensor=True)
    
    cosine_similarity = util.pytorch_cos_sim(query_embedding, context_embedding).item()
    return cosine_similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def calculate_groundedness(answer, context, model):
    """Calculates the groundedness score between an answer and context."""
    if not answer or not context:
        return 0.0

    answer_sentences = sent_tokenize(answer)
    context_sentences = sent_tokenize(context)

    answer_embeddings = model.encode(answer_sentences)
    context_embeddings = model.encode(context_sentences)

    similarity_matrix = cosine_similarity(answer_embeddings, context_embeddings)

    groundedness_scores = np.max(similarity_matrix, axis=1)

    return np.mean(groundedness_scores)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')


def calculate_completeness(context, answer, model):
    """Calculates the completeness score between a context and a generated answer."""
    if not context or not answer:
        return 0.0

    context_sentences = nltk.sent_tokenize(context)
    answer_sentences = nltk.sent_tokenize(answer)

    if not context_sentences or not answer_sentences:
        return 0.0

    context_embeddings = model.encode(context_sentences)
    answer_embeddings = model.encode(answer_sentences)

    similarity_scores = []
    for answer_embedding in answer_embeddings:
        max_similarity = 0
        for context_embedding in context_embeddings:
            similarity = cosine_similarity([answer_embedding], [context_embedding])[0][0]
            max_similarity = max(max_similarity, similarity)
        similarity_scores.append(max_similarity)

    if not similarity_scores:
        return 0.0

    return sum(similarity_scores) / len(similarity_scores)

from sentence_transformers import SentenceTransformer, util

def calculate_answer_relevancy(query, answer, model):
    """Calculates answer relevancy score based on sentence-level cosine similarity."""

    query_sentences = [query]
    answer_sentences = [s for s in answer.split('. ') if s]

    if not answer_sentences:
        return 0.0

    query_embeddings = model.encode(query_sentences)
    answer_embeddings = [model.encode(s) for s in answer_sentences]

    similarities = []
    for a_emb in answer_embeddings:
        max_sim = max([util.cos_sim(a_emb, query_embeddings).item()])
        similarities.append(max_sim)

    return sum(similarities) / len(similarities)

import pandas as pd
import os

def extract_text_from_pdf(file_path):
    """Placeholder for PDF extraction."""
    raise NotImplementedError("PDF extraction not implemented. Please use a library like PyPDF2 or pdfminer.")

def load_documents_to_dataframe(file_paths):
    """Loads documents into a pandas DataFrame."""
    data = []
    for file_path in file_paths:
        try:
            text = extract_text_from_pdf(file_path)
            title = os.path.basename(file_path)
            data.append({'document_id': file_path, 'title': title, 'text': text})
        except FileNotFoundError:
            raise
        except NotImplementedError:
             # Handle the NotImplementedError, e.g., by logging or skipping the file
            print(f"Warning: PDF extraction not implemented. Skipping {file_path}")
            
    df = pd.DataFrame(data)
    return df

def split_text_into_sentences(text):
                """Splits a text into sentences."""
                import re
                if not text:
                    return []
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
                return sentences

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np

def compute_text_projection(df, text, x, y, neighbors):
    """Computes text projection to 2D coordinates and neighbor information."""
    if df.empty:
        raise Exception("DataFrame cannot be empty.")

    if text not in df.columns:
        raise KeyError(f"Column '{text}' not found in DataFrame.")

    if not all(isinstance(item, str) for item in df[text]):
        raise TypeError("All items in the text column must be strings.")

    if df[text].isnull().any():
        raise TypeError("Text column cannot contain NaN values.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[text])

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tfidf_matrix)

    df[x] = tsne_results[:, 0]
    df[y] = tsne_results[:, 1]

    similarity_matrix = cosine_similarity(tfidf_matrix)
    neighbor_indices = np.argsort(similarity_matrix, axis=1)[:, -6:-1]  # Exclude itself

    df[neighbors] = neighbor_indices.tolist()

    return df