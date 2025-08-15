id: 689e4a3b76538d612881d1db_documentation
summary: Visual Embedding Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Visual Embedding Lab Codelab

This codelab will guide you through the QuLab application, a tool designed to explore visual embeddings of financial documents. You'll learn how to process documents, visualize their embeddings, perform semantic searches, and evaluate the quality of your results using various embedding-based metrics. This application is crucial for understanding how embeddings can represent the meaning of documents and how they can be used for effective information retrieval and analysis.

## Introduction to Embeddings and Semantic Search
Duration: 00:10

This application demonstrates the power of embeddings in understanding the relationships between text documents.  Text embeddings are vector representations of text, where similar documents are located close to each other in the vector space.  This allows for powerful semantic search capabilities.  By converting search queries into embeddings as well, we can find documents that are *semantically* similar, even if they don't share many of the same keywords.

The core concepts covered in this lab are:

*   **Document Embeddings:** Understanding how text is converted into numerical vectors representing meaning.
*   **Dimensionality Reduction:** Projecting high-dimensional embeddings into 2D space for visualization using techniques like UMAP (handled by the underlying `atlas_embed` library).
*   **Semantic Similarity:**  Using cosine similarity to measure the similarity between document embeddings.
*   **Evaluation Metrics:** Quantifying the quality of information retrieval through metrics like Context Relevancy, Groundedness, Completeness, and Answer Relevancy.

<aside class="positive">
The use of embeddings significantly enhances search capabilities, allowing for more relevant results than traditional keyword-based searches.
</aside>

## Application Architecture

The application follows a modular structure, with Streamlit handling the user interface and underlying Python libraries performing the core functionalities.

1.  **User Interface (Streamlit):** Provides interactive components for document upload, search, and result display.
2.  **Document Processing:** Extracts and cleans text from uploaded PDF documents using `pypdf`.
3.  **Embedding Generation:** Uses `sentence-transformers` to generate dense vector embeddings for each document.
4.  **Embedding Projection:** Employs `atlas_embed` (specifically `embedding_atlas.projection.compute_text_projection`) to reduce the dimensionality of the embeddings and create 2D coordinates for visualization.
5.  **Semantic Search:** Calculates cosine similarity between the query embedding and document embeddings to find the most relevant documents.
6.  **Evaluation Metrics:** Calculates and displays metrics to assess the quality of the retrieved information.
7.  **Data Storage**: Uses Streamlit's session state to persist data between page navigations.

## Document Processing
Duration: 00:15

This step involves uploading and pre-processing your financial documents. The application currently supports PDF files.

1.  **Navigate to the "Document Processing" section** using the sidebar.

2.  **Upload Financial Documents:** Use the `st.file_uploader` component to upload one or more PDF files.

    ```python
    uploaded_files = st.file_uploader("Upload Financial Documents (PDFs)", type=["pdf"], accept_multiple_files=True)
    ```

3.  **Text Extraction:** The `extract_text_from_pdfs` function extracts text from each uploaded PDF file using the `pypdf` library.

    ```python
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
    ```

4.  **Text Cleaning:** The extracted text is then cleaned using the `clean_text` function, which lowercases the text and removes extra spaces.

    ```python
    def clean_text(text):
        """Cleans text by lowercasing and removing extra spaces."""
        text = text.lower()
        text = " ".join(text.split())
        return text
    ```

5.  **Processed Documents Display:** The processed documents are displayed in a Streamlit dataframe using `st.dataframe`.

    ```python
    st.dataframe(df)
    ```

6.  **Session State:** The processed DataFrame is stored in Streamlit's session state so it can be accessed by other pages.

    ```python
    st.session_state['processed_df'] = df
    ```

## Embedding Visualization
Duration: 00:25

In this section, you'll visualize the document embeddings in an interactive atlas and perform semantic searches.

1.  **Navigate to the "Embedding Visualization" section.**

2.  **Load Processed Data:**  The application retrieves the processed DataFrame from the session state.

    ```python
    df = st.session_state['processed_df'].copy()
    ```

3.  **Embedding Generation (if needed):** If embeddings haven't already been generated, the application loads a Sentence-BERT model and generates embeddings for each document. The `sentence-transformers` library is used for this purpose.

    ```python
    model = load_sentence_bert_model()
    df = generate_embeddings(df, model)
    ```

    ```python
    def load_sentence_bert_model(model_name="all-MiniLM-L6-v2"):
        """Loads a Sentence-BERT model."""
        model = SentenceTransformer(model_name)
        return model

    def generate_embeddings(df, model):
        """Generates dense vector embeddings for each document in the DataFrame."""
        df['embeddings'] = df['text'].apply(lambda x: model.encode(str(x)) if x is not None else model.encode(""))
        return df
    ```

4.  **Embedding Projection (if needed):** If the projection hasn't been computed yet, the application uses `atlas_embed` to project the high-dimensional embeddings into a 2D space for visualization.

    ```python
    df = compute_text_projection(df, text='text', x='projection_x', y='projection_y', neighbors='neighbors')
    ```

    ```python
    def compute_text_projection(df, text, x, y, neighbors):
        """Computes a 2D projection of text embeddings for visualization."""
        atlas = AtlasEmbed(df=df, text=text)
        df = atlas.embed(x=x, y=y, neighbors=neighbors)
        return df
    ```

5.  **Interactive Embedding Atlas:** The application displays an interactive embedding atlas using the `embedding_atlas` component. The `text`, `x`, and `y` parameters specify the columns containing the document text and the x and y coordinates of the embeddings, respectively.

    ```python
    value = embedding_atlas(df, text='text', x='projection_x', y='projection_y', neighbors='neighbors', show_table=True)
    ```

6.  **Data Selection (DuckDB):** Users can select data points in the atlas, and the corresponding data is loaded from a DuckDB database.

    ```python
    predicate = value.get("predicate")
    if predicate is not None:
        try:
            selection = load_data_from_duckdb(df, predicate)
            st.subheader("Selected Data")
            st.dataframe(selection)
        except NameError as e:
            st.error(f"Error loading data from DuckDB: {e}")
    ```

    ```python
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
            selection = con.execute(f"SELECT * FROM df WHERE {predicate}\"").fetchdf()
            con.close()
            return selection
        except Exception as e:
            raise NameError(f"Error executing predicate: {e}")
    ```

7.  **Semantic Search:**  Users can enter a search query, and the application will find the most similar documents based on cosine similarity.

    ```python
    search_query = st.text_input("Enter your search query:")
    num_results = st.number_input("Number of top matches to return:", min_value=1, value=5)
    ```

    ```python
    def semantic_similarity_search(df, query_embedding, n):
        """Performs semantic similarity search to find the top-N most similar documents to a given query."""
        import numpy as np
        similarities = []
        for index, row in df.iterrows():
            doc_embedding = row['embeddings']
            similarity = compute_cosine_similarity(query_embedding, doc_embedding)
            similarities.append((row['document_id'], similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    ```

    ```python
    def compute_cosine_similarity(embedding1, embedding2):
        """Computes cosine similarity between two embeddings."""
        import numpy as np
        norm_1 = np.linalg.norm(embedding1)
        norm_2 = np.linalg.norm(embedding2)
        similarity = np.dot(embedding1, embedding2) / (norm_1 * norm_2)
        return similarity
    ```
    The cosine similarity is computed using the formula:

    $$\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} A_i^2}}$$

8. **Search Results Display:** The search results are displayed in a Streamlit dataframe, showing the document ID and similarity score.

    ```python
    search_results_df = pd.DataFrame(similarities, columns=['document_id', 'similarity_score'])
    st.dataframe(search_results_df)
    ```

<aside class="negative">
Ensure you have uploaded and processed documents in the "Document Processing" section before using the embedding visualization features.
</aside>

## Evaluation Metrics
Duration: 00:20

This section allows you to evaluate the quality of your search results by calculating various embedding-based metrics.

1.  **Navigate to the "Evaluation Metrics" section.**

2.  **Input Fields:** The application provides text areas for you to enter a query, context, and answer.

    ```python
    query = st.text_area("Query:", value="")
    context = st.text_area("Context:", value="")
    answer = st.text_area("Answer:", value="")
    ```

3.  **Metric Calculation:** The application calculates four metrics: Context Relevancy, Groundedness, Completeness, and Answer Relevancy.

    ```python
    context_relevancy = calculate_context_relevancy(query, context)
    groundedness = calculate_groundedness(answer, context)
    completeness = calculate_completeness(context, answer)
    answer_relevancy = calculate_answer_relevancy(answer, query)
    ```

    *   **Context Relevancy:** Measures the relevance of the context to the query.

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
        ```

    *   **Groundedness:** Measures the extent to which the answer is grounded in the context.

        ```python
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
        ```

    *   **Completeness:** Measures the extent to which the context covers the answer.

        ```python
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
        ```

    *   **Answer Relevancy:** Measures the relevance of the answer to the query.

        ```python
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

4.  **Metric Display:** The calculated metrics are displayed using `st.write`.

    ```python
    st.write(f"Context Relevancy: {context_relevancy:.2f}")
    st.write(f"Groundedness: {groundedness:.2f}")
    st.write(f"Completeness: {completeness:.2f}")
    st.write(f"Answer Relevancy: {answer_relevancy:.2f}")
    ```

## Conclusion
Duration: 00:05

You've now completed the QuLab codelab! You've learned how to process financial documents, visualize their embeddings in an interactive atlas, perform semantic searches, and evaluate the quality of your results.  This provides a strong foundation for understanding and utilizing text embeddings for various financial analysis tasks.  Experiment with different document sets and search queries to further explore the capabilities of the application.
