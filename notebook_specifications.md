# Technical Specification – Financial Document Similarity Atlas (Jupyter Notebook)

## 1. Overview

This notebook demonstrates how to explore semantic relationships within a corpus of financial documents using **text embeddings**, **dimensionality reduction**, and **interactive atlas-embed visualizations**.

It allows users to:

* Upload or load sample financial documents.
* View them in an interactive 2D semantic map.
* Identify topic clusters.
* Search for similar documents based on meaning rather than keywords.
* Evaluate retrieval and generation quality using embedding-based metrics.

**Learning Goals:**

* Understand how embeddings encode document meaning.
* Use `embedding_atlas.projection.compute_text_projection` to create 2D coordinates.
* Display interactive visualizations with `embedding_atlas.streamlit.embedding_atlas`.
* Perform semantic search via cosine similarity.
* Apply evaluation metrics: Context Relevancy, Groundedness, Completeness, Answer Relevancy.

---

## 2. Required Libraries

* **Core:** `pandas`, `numpy`, `scikit-learn`, `sentence_transformers`
* **PDF Extraction:** `pypdf` or `pdfminer.six`
* **Visualization:** `embedding_atlas`, `duckdb`, `streamlit` (interactive), `matplotlib`/`seaborn` (optional static)

Installation example:

```bash
pip install pandas numpy scikit-learn sentence-transformers pypdf embedding-atlas duckdb streamlit
```

---

## 3. Key Functions & Flow

### 3.1 Document Loading & Preprocessing

* Extract text from PDFs into a DataFrame:

  * `document_id`, `title`, `text` columns.
* Clean text (lowercase, remove extra spaces, normalize punctuation).

### 3.2 Embedding Generation

* Load a Sentence-BERT model (`all-MiniLM-L6-v2`).
* Generate a dense vector embedding for each document (`embeddings` column).

### 3.3 Projection for Visualization

* Use atlas-embed’s projection:

```python
from embedding_atlas.projection import compute_text_projection
compute_text_projection(df, text="text", x="projection_x", y="projection_y", neighbors="neighbors")
```

* Produces `projection_x`, `projection_y`, and `neighbors` columns for atlas rendering.

### 3.4 Interactive Visualization (Primary UI)

```python
from embedding_atlas.streamlit import embedding_atlas
value = embedding_atlas(df, text="text", x="projection_x", y="projection_y", neighbors="neighbors", show_table=True)
```

* Displays an interactive 2D atlas with optional table view.
* Returns a `predicate` string for filtering selections via DuckDB:

```python
import duckdb
predicate = value.get("predicate")
if predicate:
    selection = duckdb.query_df(df, "dataframe", f"SELECT * FROM dataframe WHERE {predicate}")
```

### 3.5 Clustering

* Apply K-Means to `projection_x` and `projection_y` → `cluster_id` column.
* View clusters interactively in atlas or as a static scatter plot.

### 3.6 Semantic Similarity Search

* Given a document ID or new query text:

  1. Generate embedding.
  2. Compute cosine similarity with all document embeddings.
  3. Return top-N matches with scores.

### 3.7 Evaluation Metrics

* **Context Relevancy:** Query ↔ Context coverage.
* **Groundedness:** Answer ↔ Context support.
* **Completeness:** Context ↔ Answer coverage.
* **Answer Relevancy:** Answer ↔ Query relevance.

All computed using sentence-level cosine similarity.

---

## 4. Outputs & Visual Components

**Required (Interactive):**

* **Document Atlas:** Interactive 2D semantic map via atlas-embed with zoom, pan, selection, and filtering.
* **Table View:** Document list with metadata and snippet preview.
* **Cluster Highlighting:** Color-coded grouping of documents in atlas.

**Optional (Static):**

* Matplotlib scatter plots for reporting.
* Histograms of similarity scores or cluster sizes.

---

## 5. Execution Flow Summary

1. **Load documents** → DataFrame (`title`, `text`).
2. **Preprocess text** for embedding.
3. **Generate embeddings** via Sentence-BERT.
4. **Compute projections** using `compute_text_projection`.
5. **Launch atlas visualization** with `embedding_atlas` in Streamlit.
6. **Cluster documents** & view grouping in atlas.
7. **Run similarity searches** to find related documents.
8. **Evaluate** with relevancy and completeness metrics.

