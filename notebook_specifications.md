
# Technical Specification for Jupyter Notebook: Financial Document Similarity Atlas

## 1. Notebook Overview

This Jupyter Notebook provides a practical case study for financial professionals to explore semantic relationships within a corpus of financial documents. Leveraging text embedding techniques and dimensionality reduction, users can visualize documents based on their semantic content, identify clusters of related information, and perform semantic similarity searches.

### Learning Goals

Upon completing this notebook, users will be able to:
*   Understand how text embeddings quantitatively capture the semantic content of financial documents.
*   Apply dimensionality reduction techniques, specifically UMAP, to visualize high-dimensional text embeddings in a 2D space for improved interpretability.
*   Identify and visualize clusters of semantically related financial documents using clustering algorithms like K-Means.
*   Perform semantic similarity searches to find documents or content highly related to a specific query or document within the corpus, utilizing cosine similarity.
*   Grasp the foundational concepts of embedding-based metrics for semantic similarity, including Context Relevancy, Groundedness, Completeness, and Answer Relevancy, as applied in document analysis.

## 2. Code Requirements

### List of Expected Libraries

The following Python libraries are expected to be used:
*   `pandas`: For data manipulation and DataFrame operations.
*   `numpy`: For numerical operations, especially with embeddings.
*   `scikit-learn`: For clustering algorithms (K-Means) and potentially other utilities.
*   `matplotlib`: For basic static plotting.
*   `seaborn`: For enhanced statistical data visualization.
*   `sentence_transformers`: For generating high-quality text embeddings.
*   `pypdf` (or `pdfminer.six`): For extracting text from PDF documents.
*   `embedding_atlas`: Specifically `embedding_atlas.projection` for dimensionality reduction.

### List of Algorithms or Functions to be Implemented

*   **Document Text Extraction**: A function to extract all text content from a PDF file.
*   **Text Preprocessing**: A function to clean raw text (e.g., lowercasing, removing extra whitespace, basic punctuation handling).
*   **Text Embedding Generation**: A function to convert text documents into numerical vector embeddings using a pre-trained Sentence-BERT model.
*   **Dimensionality Reduction**: Utilize `embedding_atlas.projection.compute_text_projection` to reduce the dimensionality of embeddings to 2D for visualization.
*   **K-Means Clustering**: A function to apply K-Means clustering to the reduced-dimension embeddings to group similar documents.
*   **Cosine Similarity Calculation**: A function to compute the cosine similarity between two given text embedding vectors or between one vector and an array of vectors.
*   **Semantic Similarity Search**: A function to find and rank documents based on their cosine similarity to a given query embedding (derived from a query string or another document's embedding).
*   **Context Relevancy Metric**: A function to calculate the context relevancy score between a query and a retrieved context based on sentence-level cosine similarity.
*   **Groundedness Metric**: A function to calculate the groundedness score between a "generated answer" and a context based on sentence-level cosine similarity.
*   **Completeness Metric**: A function to calculate the completeness score between a context and a "generated answer" based on sentence-level cosine similarity.
*   **Answer Relevancy Metric**: A function to calculate the answer relevancy score between a query and a "generated answer" based on sentence-level cosine similarity.

### Visualization like Charts, Tables, Plots

*   **Document Text Table**: A Pandas DataFrame display showing the loaded document titles and their extracted text snippets.
*   **Embedding Visualization Scatter Plot**: A 2D scatter plot of documents, where points represent documents projected into a lower-dimensional space.
*   **Clustered Embedding Visualization**: The same 2D scatter plot, but with points colored according to their assigned cluster, potentially with cluster centroids marked.
*   **Similarity Search Results Table**: A Pandas DataFrame showing the top N most similar documents, including their titles and calculated similarity scores.
*   **Histogram/Distribution Plots (Optional)**: Visualizations to understand the distribution of similarity scores or cluster sizes.

## 3. Notebook Sections (in detail)

The notebook will be structured into 18 main sections, each comprising Markdown explanations and corresponding Code cells for implementation and execution.

---

### Section 1: Introduction to Financial Document Analysis

**Markdown Cell:**
Explain the challenge of understanding large volumes of financial documents and how this notebook will demonstrate leveraging NLP techniques to address it. Introduce the concept of semantic similarity and its application in financial analysis for tasks like competitive analysis, research, and trend identification.

### Section 2: Setup and Library Installation

**Markdown Cell:**
Detail the necessary Python libraries for this project. Explain their purpose (e.g., `pandas` for data handling, `sentence_transformers` for embeddings, `embedding_atlas` for projection, `pypdf` for PDF reading, `matplotlib`/`seaborn` for visualization).

**Code Cell (Implementation):**
```python
# Install necessary libraries
# Implement pip install commands for all required libraries
```

**Code Cell (Execution):**
```python
# Execute the installation commands
```

**Markdown Cell:**
Confirm that all required libraries have been successfully installed and are ready for use.

---

### Section 3: Data Acquisition and Document Loading

**Markdown Cell:**
Explain that the application allows users to upload financial documents. For demonstration, this notebook will use a default sample PDF. Instruct the user to place a file named `sample_financial_report.pdf` (e.g., an annual report, research paper, or earnings call transcript) in the same directory as the notebook. Describe the process of loading this PDF and extracting its text content.

**Code Cell (Implementation):**
```python
# Import necessary libraries for PDF processing (e.g., PyPDF2 or pdfminer.six)
# Define a function to extract text from a PDF file given its path.
# This function should iterate through pages and extract text, then return a single string.

# Define a function to load documents into a pandas DataFrame.
# This function will take a list of file paths (or a directory path)
# Extract text for each document and store it in a DataFrame with columns like 'document_id', 'title', 'text'.
# For demonstration, hardcode the path to the sample PDF.
```

**Code Cell (Execution):**
```python
# Specify the path to the sample PDF document.
# Load the sample PDF document into a DataFrame using the defined function.
# Display the head of the DataFrame to show loaded documents.
```

**Markdown Cell:**
Explain the structure of the loaded data and confirm successful text extraction from the sample document.

---

### Section 4: Text Preprocessing

**Markdown Cell:**
Discuss the importance of text preprocessing to clean raw textual data before generating embeddings. Explain common steps like lowercasing, removing extra whitespace, and potentially removing special characters or numbers to standardize the text and improve embedding quality.

**Code Cell (Implementation):**
```python
# Define a text cleaning function (e.g., convert to lowercase, remove multiple spaces, remove newlines).
# This function should take a string and return a cleaned string.
```

**Code Cell (Execution):**
```python
# Apply the text cleaning function to the 'text' column of the DataFrame.
# Display a sample of the cleaned text.
```

**Markdown Cell:**
Illustrate the impact of preprocessing on the text and explain why these steps are beneficial for subsequent analyses.

---

### Section 5: Text Embedding Generation

**Markdown Cell:**
Introduce the concept of text embeddings as numerical vector representations that capture the semantic meaning of text. Explain that documents with similar meanings will have embeddings that are close to each other in the vector space. Briefly mention Sentence-BERT models as an effective approach for generating such embeddings, which capture fine-grained semantic similarities, as highlighted in the provided document. We will use a pre-trained model for this purpose.

**Code Cell (Implementation):**
```python
# Import SentenceTransformer from sentence_transformers.
# Instantiate a pre-trained Sentence-BERT model (e.g., 'all-MiniLM-L6-v2').
# Define a function to generate embeddings for a list of texts.
```

**Code Cell (Execution):**
```python
# Generate embeddings for the 'cleaned_text' column of the DataFrame.
# Store these embeddings in a new column (e.g., 'embeddings').
# Display the shape of the embeddings array.
```

**Markdown Cell:**
Explain that each document is now represented by a high-dimensional vector and reiterate that these vectors encode the semantic information of the text.

---

### Section 6: Dimensionality Reduction for Visualization (UMAP)

**Markdown Cell:**
Explain that text embeddings are high-dimensional, making direct visualization challenging. Introduce dimensionality reduction techniques like UMAP (Uniform Manifold Approximation and Projection) which project these high-dimensional vectors into a lower-dimensional space (typically 2D) while preserving the essential semantic relationships and local/global structure. This allows for visual interpretation of document similarity. The formula for cosine similarity, which underpins these semantic relationships, is given by:
$$ \text{Sim}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||} $$
where $\mathbf{A}$ and $\mathbf{B}$ are two embedding vectors, $\mathbf{A} \cdot \mathbf{B}$ is their dot product, and $||\mathbf{A}||$ and $||\mathbf{B}||$ are their Euclidean norms. Cosine similarity ranges from -1 (opposite) to 1 (identical).

**Code Cell (Implementation):**
```python
# Import compute_text_projection from embedding_atlas.projection.
# Define a function to apply dimensionality reduction and get 2D coordinates.
# This function will use compute_text_projection.
```

**Code Cell (Execution):**
```python
# Apply the dimensionality reduction function to the DataFrame's 'embeddings' column.
# Store the projected 2D coordinates in 'projection_x' and 'projection_y' columns.
# Display the head of the DataFrame with the new projection columns.
```

**Markdown Cell:**
Explain that `projection_x` and `projection_y` now represent the 2D coordinates for each document, ready for plotting.

---

### Section 7: Embedding Visualization

**Markdown Cell:**
Describe the purpose of visualizing the 2D embeddings: to visually inspect how documents are semantically related. Documents that are semantically similar should appear closer to each other on the plot.

**Code Cell (Implementation):**
```python
# Import matplotlib.pyplot and seaborn.
# Define a function to create a scatter plot of the projected embeddings.
# The plot should use 'projection_x' and 'projection_y'.
# Add labels for axes and a title.
# (Optional: Add basic annotations for a few document titles/indices to demonstrate identity)
```

**Code Cell (Execution):**
```python
# Execute the plotting function using the DataFrame.
# Display the plot.
```

**Markdown Cell:**
Interpret the initial visualization, pointing out visible clusters or relationships and setting the stage for formal clustering.

---

### Section 8: Clustering Document Embeddings

**Markdown Cell:**
Explain that while the visualization shows potential groupings, clustering algorithms can formally identify these semantically related groups. Introduce K-Means clustering as a simple and effective algorithm that partitions data into a predefined number of clusters by minimizing within-cluster variance. This allows us to define "strata" or topics within the document collection.

**Code Cell (Implementation):**
```python
# Import KMeans from sklearn.cluster.
# Define a function to apply K-Means clustering to the 2D projected embeddings.
# The function should take the DataFrame and the desired number of clusters (e.g., 5 or 7).
# It should return the cluster labels.
```

**Code Cell (Execution):**
```python
# Define a suitable number of clusters (e.g., 5).
# Apply the K-Means clustering function to the projected embeddings.
# Add the resulting cluster labels as a new column (e.g., 'cluster_id') to the DataFrame.
# Display the value counts for the 'cluster_id' column.
```

**Markdown Cell:**
Explain what the `cluster_id` column represents and how these clusters will be visualized.

---

### Section 9: Visualizing Clustered Documents

**Markdown Cell:**
Explain how coloring the embedding plot by cluster ID can reveal the natural groupings of semantically related documents. Each cluster represents a distinct topic or theme within the financial document corpus.

**Code Cell (Implementation):**
```python
# Import matplotlib.pyplot and seaborn.
# Define a function to create a scatter plot of the projected embeddings,
# with points colored by 'cluster_id'.
# Use a distinct colormap for clusters.
# Add a legend for cluster IDs.
# Add labels for axes and a title.
```

**Code Cell (Execution):**
```python
# Execute the plotting function with cluster IDs.
# Display the plot.
```

**Markdown Cell:**
Analyze the clustered visualization. Discuss how documents within the same cluster likely cover similar financial topics or themes, and how distant clusters represent distinct content areas.

---

### Section 10: Semantic Similarity Search: Concept

**Markdown Cell:**
Introduce semantic similarity search as a core feature for finding related financial content. Explain that by converting text into embeddings, we can quantify how similar two pieces of text are using metrics like cosine similarity. A higher cosine similarity score indicates greater semantic closeness.
The formula for cosine similarity between two embedding vectors $\mathbf{A}$ and $\mathbf{B}$ is:
$$ \text{Sim}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||} $$
where $\mathbf{A} \cdot \mathbf{B}$ is the dot product of the vectors, and $||\mathbf{A}||$ and $||\mathbf{B}||$ are their Euclidean norms.

---

### Section 11: Semantic Similarity Search: Implementation

**Markdown Cell:**
Describe the function to perform a semantic similarity search. This function will take a query (either a document ID from the corpus or a new text string), convert it to an embedding, and then compute its cosine similarity with all other document embeddings in the corpus. Finally, it will return the top N most similar documents.

**Code Cell (Implementation):**
```python
# Import cosine_similarity from sklearn.metrics.pairwise.
# Define a function `find_similar_documents` that takes:
#   - the DataFrame with 'embeddings'
#   - a query (can be a document ID or a string)
#   - the Sentence-BERT model for new query embedding
#   - an integer N for the number of top similar documents to return
# The function should:
#   1. Get the embedding for the query (either from DataFrame or generate it if it's a new string).
#   2. Compute cosine similarity between the query embedding and all document embeddings.
#   3. Sort documents by similarity score in descending order.
#   4. Return the top N documents and their scores.
```

**Code Cell (Execution):**
```python
# Select a sample document by its 'document_id' from the loaded DataFrame to use as a query.
# Or, define a new text query string related to financial topics (e.g., "annual revenue growth for technology companies").
# Execute the `find_similar_documents` function with the chosen query and N=5.
# Display the results in a readable format (e.g., document title and similarity score).
```

**Markdown Cell:**
Interpret the results of the similarity search, highlighting how relevant documents are returned based on their semantic content, even if they don't share exact keywords.

---

### Section 12: Advanced Functionality Metrics Overview

**Markdown Cell:**
Explain that beyond simple similarity, evaluating language models in applications like Retrieval-Augmented Generation (RAG) systems in finance requires more nuanced metrics. Introduce the concepts of Context Relevancy, Groundedness, Completeness, and Answer Relevancy, as discussed in the provided document. These embedding-based metrics offer transparent and interpretable insights into model performance. For the purpose of this notebook, we will simulate these by taking predefined "query", "context", and "answer" sentences from our documents.

---

### Section 13: Context Relevancy Metric

**Markdown Cell:**
Define **Context Relevancy** as a measure of how well retrieved documents address an input query. This is quantified by sentence-level semantic similarity. Given a query $Q = \{q_1, q_2, \dots, q_m\}$ and a retrieved context $C = \{c_1, c_2, \dots, c_n\}$, the maximum similarity for each query sentence $q_i$ with any context sentence $c_j$ is:
$$ S_{\text{max}}(q_i) = \max_{1 \le j \le n} \text{Sim}(q_i, c_j) $$
The overall context relevancy score is then aggregated by averaging these maximum similarities:
$$ S_{\text{c-relevancy}} = \frac{1}{m} \sum_{i=1}^m S_{\text{max}}(q_i) $$
A high score indicates that the context is highly relevant to the query.

**Code Cell (Implementation):**
```python
# Define a helper function to split text into sentences (e.g., using NLTK or simple regex).
# Define a function `calculate_context_relevancy` that takes:
#   - a query string
#   - a context string
#   - the Sentence-BERT model
# The function should:
#   1. Split query and context into sentences.
#   2. Generate embeddings for all query and context sentences.
#   3. Compute pairwise cosine similarities between query sentences and context sentences.
#   4. Calculate S_max(q_i) for each query sentence.
#   5. Calculate the average of S_max(q_i) for the final S_c-relevancy score.
```

**Code Cell (Execution):**
```python
# Select a portion of text from one document as a "query" and a portion from another as "context".
# For demonstration, manually extract relevant sentences.
query_text = "What is the company's financial performance?"
context_text = "In the fiscal year, our revenue increased by 15% to $500 million, driven by strong sales in digital services. Net income reached $75 million, showing healthy profitability."
# Execute `calculate_context_relevancy` with these sample texts.
# Print the calculated context relevancy score.
```

**Markdown Cell:**
Explain the calculated score and its implications regarding how well the context addresses the query.

---

### Section 14: Groundedness Metric

**Markdown Cell:**
Define **Groundedness** as ensuring that generated content is directly supported by the retrieved documents, preventing unsupported statements or "hallucinations". Given a "generated answer" $A = \{a_1, a_2, \dots, a_k\}$ and a context $C = \{c_1, c_2, \dots, c_n\}$, the maximum similarity for each answer sentence $a_i$ with any context sentence $c_j$ is:
$$ S_{\text{max}}(a_i) = \max_{1 \le j \le n} \text{Sim}(a_i, c_j) $$
The overall groundedness score is aggregated by averaging these maximum similarities:
$$ S_{\text{groundedness}} = \frac{1}{k} \sum_{i=1}^k S_{\text{max}}(a_i) $$
A high score indicates that the answer is well-supported by the context.

**Code Cell (Implementation):**
```python
# Define a function `calculate_groundedness` that takes:
#   - an answer string
#   - a context string
#   - the Sentence-BERT model
# The function should:
#   1. Split answer and context into sentences.
#   2. Generate embeddings for all answer and context sentences.
#   3. Compute pairwise cosine similarities between answer sentences and context sentences.
#   4. Calculate S_max(a_i) for each answer sentence.
#   5. Calculate the average of S_max(a_i) for the final S_groundedness score.
```

**Code Cell (Execution):**
```python
# Select a portion of text from a document as "context" and craft a "generated answer"
# that is mostly, but not entirely, supported by the context to demonstrate grounding.
context_text = "The company reported a 10% increase in Q3 revenue. Operating expenses remained stable. Our CEO stated optimism for the next quarter."
answer_text = "The company's revenue grew by 10% in Q3, and expenses did not change. The CEO is very confident about future growth, expecting 20% increase." # Slight hallucination
# Execute `calculate_groundedness` with these sample texts.
# Print the calculated groundedness score.
```

**Markdown Cell:**
Discuss the calculated groundedness score, explaining if the "answer" is well-supported or if there are signs of ungrounded information (hallucinations).

---

### Section 15: Completeness Metric

**Markdown Cell:**
Define **Completeness** as evaluating whether the "generated answer" covers all relevant information from the context. Given a context $C = \{c_1, c_2, \dots, c_n\}$ and a "generated answer" $A = \{a_1, a_2, \dots, a_k\}$, the maximum similarity for each context sentence $c_i$ with any answer sentence $a_j$ is:
$$ S_{\text{max}}(c_i) = \max_{1 \le j \le k} \text{Sim}(c_i, a_j) $$
The overall completeness score is aggregated by averaging these maximum similarities:
$$ S_{\text{completeness}} = \frac{1}{n} \sum_{i=1}^n S_{\text{max}}(c_i) $$
A high score indicates that the answer covers most of the content from the context.

**Code Cell (Implementation):**
```python
# Define a function `calculate_completeness` that takes:
#   - a context string
#   - an answer string
#   - the Sentence-BERT model
# The function should:
#   1. Split context and answer into sentences.
#   2. Generate embeddings for all context and answer sentences.
#   3. Compute pairwise cosine similarities between context sentences and answer sentences.
#   4. Calculate S_max(c_i) for each context sentence.
#   5. Calculate the average of S_max(c_i) for the final S_completeness score.
```

**Code Cell (Execution):**
```python
# Select a portion of text from a document as "context" and craft a "generated answer"
# that is either very complete or intentionally incomplete to demonstrate.
context_text = "The Q4 earnings report highlighted significant growth in the cloud computing division. However, the traditional software sales saw a slight decline. International markets performed strongly."
answer_text = "Q4 earnings showed strong growth in cloud computing. International markets performed well." # Incomplete
# Execute `calculate_completeness` with these sample texts.
# Print the calculated completeness score.
```

**Markdown Cell:**
Discuss the calculated completeness score, explaining if the "answer" adequately covers the information in the context or if significant portions are missing.

---

### Section 16: Answer Relevancy Metric

**Markdown Cell:**
Define **Answer Relevancy** as ensuring that the "generated response" directly addresses the user's query. Given a "generated answer" $A = \{a_1, a_2, \dots, a_k\}$ and a query $Q = \{q_1, q_2, \dots, q_m\}$, the maximum similarity for each answer sentence $a_i$ with any query sentence $q_j$ is:
$$ S_{\text{max}}(a_i) = \max_{1 \le j \le m} \text{Sim}(a_i, q_j) $$
The overall answer relevancy score is aggregated by averaging these maximum similarities:
$$ S_{\text{a-relevancy}} = \frac{1}{k} \sum_{i=1}^k S_{\text{max}}(a_i) $$
A high score indicates that the answer effectively addresses the user's question.

**Code Cell (Implementation):**
```python
# Define a function `calculate_answer_relevancy` that takes:
#   - a query string
#   - an answer string
#   - the Sentence-BERT model
# The function should:
#   1. Split query and answer into sentences.
#   2. Generate embeddings for all query and answer sentences.
#   3. Compute pairwise cosine similarities between answer sentences and query sentences.
#   4. Calculate S_max(a_i) for each answer sentence.
#   5. Calculate the average of S_max(a_i) for the final S_a-relevancy score.
```

**Code Cell (Execution):**
```python
# Define a query string.
# Craft a "generated answer" that is either highly relevant or somewhat irrelevant.
query_text = "What were the key takeaways from the earnings call regarding future investments?"
answer_text = "The earnings call emphasized strategic investments in AI research and development. The CEO also mentioned expanding into new geographical markets." # Highly relevant
# Execute `calculate_answer_relevancy` with these sample texts.
# Print the calculated answer relevancy score.
```

**Markdown Cell:**
Discuss the calculated answer relevancy score, explaining how well the "answer" aligns with the user's intent.

---

### Section 17: Discussion of Advanced Metrics & Limitations

**Markdown Cell:**
Briefly discuss that the presented functionality metrics (Context Relevancy, Groundedness, Completeness, Answer Relevancy) are foundational. Acknowledge the existence of more advanced techniques mentioned in the provided document, such as Natural Language Inference (NLI) based groundedness, Wasserstein distance for completeness, and the crucial aspects of Human-Calibrated Automated Testing (HCAT) including probability calibration and conformal prediction for robustness and weakness identification. Explain that these advanced methods, while powerful for real-world RAG validation in high-stakes environments, are beyond the scope of this introductory notebook but are critical for comprehensive GLM evaluation.

---

### Section 18: Conclusion

**Markdown Cell:**
Summarize the key achievements of this notebook: demonstrating how text embeddings, dimensionality reduction, clustering, and semantic similarity calculations can be used to analyze financial documents. Reiterate the learning outcomes achieved and emphasize the value of such tools for financial professionals in understanding large document corpuses. Briefly touch upon potential future work, such as integrating these analyses into a full RAG system pipeline, exploring different embedding models, or incorporating advanced validation metrics.

---
