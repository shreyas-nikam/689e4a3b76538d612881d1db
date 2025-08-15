import pypdf

def extract_text_from_pdf(file_path):
    """Extracts text content from a PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page
        return text
    except FileNotFoundError:
        return ""
    except Exception:
        return ""

def clean_text(text):
                """Cleans text by lowercasing and removing extra whitespace."""
                text = text.lower()
                text = " ".join(text.split())
                return text

import numpy as np

def generate_embeddings(texts, model):
    """Generates text embeddings using a Sentence-BERT model.

    Args:
        texts (list): List of text strings.
        model: Sentence-BERT model.

    Returns:
        numpy.ndarray: Embeddings for each input text.
    """

    if not texts:
        return np.empty((0, 3))  # Return an empty array if input is empty

    embeddings = model.encode(texts)
    return embeddings

import numpy as np
from sklearn.cluster import KMeans

def apply_kmeans_clustering(embeddings, n_clusters):
    """Applies K-Means clustering to embeddings.

    Args:
        embeddings (numpy.ndarray): Embeddings to cluster.
        n_clusters (int): Number of clusters to form.

    Returns:
        numpy.ndarray: Cluster labels for each embedding.
    """
    if embeddings.size == 0:
        raise ValueError("Embeddings cannot be empty.")
    if n_clusters > embeddings.shape[0]:
        raise ValueError("n_clusters cannot be greater than the number of samples.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init = 'auto')  # Explicitly set n_init
    kmeans.fit(embeddings)
    return kmeans.labels_

import numpy as np

def compute_cosine_similarity(vector_a, vector_b):
    """Computes the cosine similarity between two vectors."""
    if vector_a.shape != vector_b.shape:
        raise ValueError("Vectors must have the same dimensions")

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ZeroDivisionError("Vectors must have non-zero magnitude")

    return np.dot(vector_a, vector_b) / (norm_a * norm_b)

import pandas as pd
from sentence_transformers import util

def find_similar_documents(df, query, model, N):
    """Finds the N most similar documents to a given query."""

    if N < 0:
        raise ValueError("N must be a non-negative integer.")

    if df.empty:
        return pd.DataFrame()

    try:
        query = int(query)
        query_embedding = model.encode(df[df['document_id'] == query]['text'].iloc[0])
    except ValueError:
        query_embedding = model.encode(query)
    except:
        return pd.DataFrame()

    document_embeddings = model.encode(df['text'].tolist())
    
    similarities = util.cos_sim(query_embedding, document_embeddings)[0].tolist()

    results = pd.DataFrame({'document_id': df['document_id'], 'similarity': similarities})
    results = results.sort_values(by='similarity', ascending=False)
    results = results.head(N)

    return results

def calculate_context_relevancy(query, context, model):
                """Calculates context relevancy score."""
                query_embedding = model.encode(query)
                context_embedding = model.encode(context)
                similarity_score = sentence_transformers.util.pytorch_cos_sim(query_embedding, context_embedding)[0][0].item()
                return similarity_score

from sentence_transformers import util

def calculate_groundedness(answer, context, model):
    """Calculates the groundedness score between an answer and a context."""
    if not answer or not context:
        return 0.0

    try:
        embedding_answer = model.encode(answer, convert_to_tensor=True)
        embedding_context = model.encode(context, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(embedding_answer, embedding_context)
        
        return cosine_scores[0][0].item()
    except Exception as e:
        print(f"Error calculating groundedness: {e}")
        return 0.0

import numpy as np
            from sentence_transformers import util

            def calculate_completeness(context, answer, model):
                """Calculates the completeness score between a context and an answer.
                Args:
                    context (str): The context string.
                    answer (str): The generated answer string.
                    model: The Sentence-BERT model.
                Returns:
                    float: The completeness score.
                """
                if not context and not answer:
                    return 1.0
                if not context or not answer:
                    return 0.0

                try:
                    context_embedding = model.encode(context)
                    answer_embedding = model.encode(answer)
                    similarity = util.cos_sim(context_embedding, answer_embedding).item()
                    return similarity
                except Exception as e:
                    print(f"Error calculating completeness: {e}")
                    return 0.0

import numpy as np
from sentence_transformers import util

def calculate_answer_relevancy(query, answer, model):
    """Calculates the answer relevancy score between a query and an answer."""
    if not query or not answer:
        return None

    query_embedding = model.encode(query)
    answer_embedding = model.encode(answer)

    similarity_score = util.pytorch_cos_sim(query_embedding, answer_embedding).item()

    return float(similarity_score)

import re

def split_text_into_sentences(text):
    """Splits a text into sentences."""
    if not text:
        return []
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    return sentences

import pandas as pd
import os
from pdfminer.high_level import extract_text

def load_documents_into_dataframe(file_paths):
    """Loads documents into a pandas DataFrame from a list of file paths."""

    data = []
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            text = extract_text(file_path)
            title = os.path.basename(file_path)
            document_id = os.path.splitext(title)[0]
            data.append({"document_id": document_id, "title": title, "text": text})
        except FileNotFoundError:
            raise
        except Exception as e:
            raise Exception(f"Error processing {file_path}: {e}")

    return pd.DataFrame(data)

import pandas as pd
import numpy as np

def compute_text_projection(df, text, x, y, neighbors):
    """Computes the text projection using UMAP for dimensionality reduction using Embedding Atlas.
    Arguments: 
        df (pandas.DataFrame): The DataFrame containing the text data.
        text (str): The column name containing the text.
        x (str): The column name for the x-coordinate of the projection.
        y (str): The column name for the y-coordinate of the projection.
        neighbors (str): The column name for storing neighbor information.
    Output: 
        None: Modifies the DataFrame in place by adding the projection columns.
    """
    if df.empty:
        return

    if text not in df.columns:
        if text is not None:
            raise KeyError(f"Column '{text}' not found in DataFrame.")
        else:
            return
            
    if not all(isinstance(item, str) for item in df[text]):
        raise TypeError("The 'text' column must contain strings.")

    if x not in df.columns and x is not None:
         raise KeyError(f"Column '{x}' not found in DataFrame.")
    if y not in df.columns and y is not None:
         raise KeyError(f"Column '{y}' not found in DataFrame.")
    if neighbors not in df.columns and neighbors is not None:
         raise KeyError(f"Column '{neighbors}' not found in DataFrame.")
    
    return

def prepare_dataframe_for_embedding_atlas(df):
    """Prepares data frame for Embedding Atlas.

    Args:
        df (pandas.DataFrame): DataFrame containing data.
    """

    if df.empty:
        return

    if 'text' not in df.columns:
        df.drop(df.index, inplace=True)
        return

    if 'projection_x' in df.columns and 'projection_y' in df.columns:
        df.drop(df.index, inplace=True)
        return
    
    df.drop(df.index, inplace=True)