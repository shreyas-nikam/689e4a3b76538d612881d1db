
## Streamlit Application Requirements Specification

### 1. Application Overview

This Streamlit application aims to provide an interactive tool for exploring semantic relationships within a corpus of financial documents. It allows users to upload documents, visualize their embeddings in an interactive atlas, perform semantic searches, and evaluate retrieval quality using embedding-based metrics.

**Learning Goals:**

*   Understand how embeddings encode document meaning.
*   Use `embedding_atlas.projection.compute_text_projection` to create 2D coordinates for visualization.
*   Display interactive visualizations with `embedding_atlas.streamlit.embedding_atlas`.
*   Perform semantic search via cosine similarity.
*   Apply various embedding-based evaluation metrics: Context Relevancy, Groundedness, Completeness, and Answer Relevancy.

### 2. User Interface Requirements

#### Layout and Navigation Structure

The application will consist of the following sections:

1.  **File Upload:**  Allows the user to upload financial documents (PDFs). A default sample PDF document will be provided for demonstration purposes.
2.  **Data Processing:**  Displays the uploaded documents in a table format.
3.  **Embedding Visualization:** Show an interactive 2D semantic map of the documents, allowing users to explore relationships and clusters.
4.  **Semantic Search:**  Enables users to enter a search query and find similar documents.
5.  **Evaluation Metrics:**  Displays calculated evaluation metrics for a given query, context, and answer.

#### Input Widgets and Controls

*   **File Uploader:** For uploading financial documents (PDFs).
*   **Text Input:** For entering search queries.
*   **Slider/Number Input:** To define the number of clusters for K-means clustering.
*   **Number Input:** For defining the number of top matches for semantic similarity search.
*   **Text area**: User to enter Query, context and answer to test the Evaluation Metrics
*   **Buttons:** To trigger actions such as "Compute Embeddings", "Generate Projection", "Perform Search", "Calculate Metrics".

#### Visualization Components

*   **Interactive Embedding Atlas:**  Uses `embedding_atlas.streamlit.embedding_atlas` to display the 2D semantic map.
*   **Data Tables:** To display the uploaded documents, search results, and evaluation metrics.
*   **Scatter Plot:** For visualizing the clustered documents with cluster IDs.

#### Interactive Elements and Feedback Mechanisms

*   **Tooltips/Annotations:** Display document titles and other relevant information on hover in the embedding atlas.
*   **Dynamic Filtering:** The embedding atlas allows users to select documents, and the application should dynamically filter the displayed data based on the selection.
*   **Search Results Highlighting:**  Highlight the search results in the embedding atlas and the data tables.
*   **Status Messages:** Display messages to indicate the progress of data processing, embedding generation, etc.

### 3. Additional Requirements

*   **Annotation and Tooltip Specifications:**
    *   Hovering over a point in the Embedding Atlas should display the document title, document ID, and a snippet of the document text.
    *   Clicking on a point should highlight the corresponding row in the data table.
*   **State Management:**
    *   Use `streamlit.session_state` to preserve the uploaded documents, generated embeddings, projection coordinates, and clustering results between interactions.

### 4. Notebook Content and Code Requirements

#### 4.1. Document Loading & Preprocessing

**Code:**
```python
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
```

**Usage:**

*   Use `streamlit.file_uploader` to allow users to upload PDF files.
*   Store the uploaded files in `streamlit.session_state`.
*   Call `extract_text_from_pdfs` to extract the text from the PDFs.
*   Call `clean_text` to preprocess the extracted text.
*   Display the processed data in a `streamlit.dataframe`.

**Markdown Content:**
```markdown
### Document Loading & Preprocessing

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]
```

#### 4.2. Embedding Generation

**Code:**
```python
from sentence_transformers import SentenceTransformer

def load_sentence_bert_model(model_name="all-MiniLM-L6-v2"):
    """Loads a Sentence-BERT model."""
    model = SentenceTransformer(model_name)
    return model

def generate_embeddings(df, model):
    """Generates dense vector embeddings for each document in the DataFrame."""
    df['embeddings'] = df['text'].apply(lambda x: model.encode(str(x)) if x is not None else model.encode(""))
    return df
```

**Usage:**

*   Load the Sentence-BERT model using `load_sentence_bert_model`.
*   Call `generate_embeddings` to generate embeddings for the cleaned text.
*   Store the DataFrame with embeddings in `streamlit.session_state`.

**Mathematical Concept (LaTeX):**

An embedding $E(T)$ for a piece of text $T$ is a vector in $\mathbb{R}^d$, where $d$ is the dimensionality of the embedding space. The key property is that the geometric proximity of these vectors corresponds to semantic similarity. That is, if text $T_1$ is semantically similar to text $T_2$, then the Euclidean distance or cosine distance between their embeddings $E(T_1)$ and $E(T_2)$ will be small.

**Cosine Similarity ($cos(\theta)$):**

$$ \text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}} $$

Where:

*   $A$ and $B$ are the embedding vectors of two documents.
*   $A \cdot B$ is the dot product of $A$ and $B$.
*   $\|A\|$ and $\|B\|$ are the Euclidean norms (magnitudes) of vectors $A$ and $B$.

**Markdown Content:**

```markdown
### Embedding Generation

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]

```

#### 4.3. Projection for Visualization

**Code:**
```python
from atlas_embed import AtlasEmbed

def compute_text_projection(df, text, x, y, neighbors):
    """Computes a 2D projection of text embeddings for visualization."""
    atlas = AtlasEmbed(df=df, text=text)
    df = atlas.embed(x=x, y=y, neighbors=neighbors)
    return df
```

**Usage:**

*   Call `compute_text_projection` to generate the 2D projection coordinates.
*   Store the DataFrame with projection coordinates in `streamlit.session_state`.

**Conceptual Formula (LaTeX):**

$$ (x_i, y_i) = f(E_i) $$

Where:

*   $E_i$ is the high-dimensional embedding vector for document $i$.
*   $(x_i, y_i)$ are the 2D coordinates for document $i$ in the projected space.

**Markdown Content:**
```markdown
### Projection for Visualization

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]
```

#### 4.4. Interactive Visualization

**Code:**

```python
import pandas as pd
import duckdb
from embedding_atlas.streamlit import embedding_atlas

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
```

**Usage:**

*   Call `embedding_atlas` to display the interactive visualization.
*   Use the returned predicate string to filter the DataFrame using DuckDB.
*   Display the filtered data in a `streamlit.dataframe`.

**Markdown Content:**
```markdown
### Interactive Visualization (Primary UI)

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]
```

#### 4.5. Clustering

**Code:**
```python
from sklearn.cluster import KMeans

def apply_kmeans_clustering(df, n_clusters, x_col, y_col):
    """Applies K-Means clustering to the projection coordinates."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df['cluster_id'] = kmeans.fit_predict(df[[x_col, y_col]])
    return df
```

**Usage:**

*   Get the number of clusters from a `streamlit.number_input`.
*   Call `apply_kmeans_clustering` to apply K-Means clustering to the projection coordinates.
*   Store the DataFrame with cluster IDs in `streamlit.session_state`.
*   Visualize the clustered data using `streamlit.scatter_chart` or by color-coding the points in the `embedding_atlas`.

**Mathematical Concept (LaTeX):**

$$ \text{WCSS} = \sum_{j=1}^{k} \sum_{i \in S_j} \|x_i - \mu_j\|^2 $$

Where:

*   $k$ is the number of clusters.
*   $S_j$ is the set of data points belonging to cluster $j$.
*   $x_i$ is a data point.
*   $\mu_j$ is the centroid (mean) of cluster $j$.

**Markdown Content:**
```markdown
### Clustering

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]
```

#### 4.6. Semantic Similarity Search

**Code:**
```python
import numpy as np

def compute_cosine_similarity(embedding1, embedding2):
    """Computes cosine similarity between two embeddings."""
    norm_1 = np.linalg.norm(embedding1)
    norm_2 = np.linalg.norm(embedding2)

    if norm_1 == 0 or norm_2 == 0:
        return np.nan

    similarity = np.dot(embedding1, embedding2) / (norm_1 * norm_2)
    return similarity

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
        # Ensure both are numpy arrays for dot product and norm
        if not isinstance(doc_embedding, np.ndarray):
            doc_embedding = np.array(doc_embedding)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        similarity = compute_cosine_similarity(query_embedding, doc_embedding)
        similarities.append((row['document_id'], similarity))

    # Filter out NaN similarities if any (due to zero norm embeddings)
    similarities = [(doc_id, score) for doc_id, score in similarities if not np.isnan(score)]

    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:min(n, len(similarities))]
```

**Usage:**

*   Get the search query from a `streamlit.text_input`.
*   Encode the search query using the Sentence-BERT model.
*   Call `semantic_similarity_search` to find similar documents.
*   Display the search results in a `streamlit.dataframe`.

**Mathematical Formula (LaTeX):**

$$\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Markdown Content:**
```markdown
### Semantic Similarity Search

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]
```

#### 4.7. Evaluation Metrics

**Code:**
```python
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
```

**Usage:**

*   Get the query, context and answer from `streamlit.text_area`.
*   Call the evaluation metrics calculations functions.
*   Display the evaluation metrics to the user

**Mathematical Concept (LaTeX):**

*   Context Relevancy:
     $$ \text{Context Relevancy} = \frac{|\text{words in query} \cap \text{words in context}|}{|\text{words in query}|} $$
*   Groundedness:
     $$ \text{Groundedness} = \frac{|\text{words in answer} \cap \text{words in context}|}{|\text{words in answer}|} $$
*   Completeness:
     $$ \text{Completeness} = \frac{|\text{words in context} \cap \text{words in answer}|}{|\text{words in context}|} $$
*   Answer Relevancy:
     $$ \text{Answer Relevancy} = \frac{|\text{words in answer} \cap \text{words in query}|}{|\text{words in query}|} $$

**Markdown Content:**
```markdown
### Evaluation Metrics

**Business Value:**
[As detailed in the notebook markdown.]

**Technical Implementation:**
[As detailed in the notebook markdown.]
```
