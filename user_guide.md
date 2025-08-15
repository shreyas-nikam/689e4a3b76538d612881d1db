id: 689e4a3b76538d612881d1db_user_guide
summary: Visual Embedding User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Visual Embedding Lab User Guide

This codelab guides you through the QuLab application, which is designed to help you understand and explore visual embeddings of financial documents. By the end of this guide, you'll be able to upload documents, visualize their embeddings in an interactive atlas, perform semantic searches, and evaluate retrieval quality using various embedding-based metrics. This lab will help you understand how embeddings encode document meaning and how they can be used for semantic search and evaluation.

## Introduction to QuLab

Duration: 00:05

QuLab provides a hands-on environment to explore the power of embeddings in understanding and analyzing financial documents. Embeddings are numerical representations of text that capture their semantic meaning. In this lab, you'll learn how to use these embeddings to visualize relationships between documents, search for similar content, and evaluate the quality of search results. This is particularly useful in finance for tasks like identifying related news articles, finding relevant documents for regulatory compliance, and building intelligent search systems.

<aside class="positive">
<b>Tip:</b> Understanding embeddings is crucial for modern NLP applications, especially in finance where large volumes of textual data need to be analyzed efficiently.
</aside>

## Document Processing

Duration: 00:10

This section focuses on how to upload and preprocess financial documents using the QuLab application.

1.  **Accessing Document Processing:** Navigate to the "Document Processing" section using the sidebar on the left.
2.  **Uploading Documents:** Click on the "Upload Financial Documents (PDFs)" button to upload PDF files from your local machine. You can upload multiple files at once.
3.  **Viewing Processed Documents:** Once the files are uploaded, the application extracts the text from each PDF, cleans it by lowercasing and removing extra spaces, and displays the processed documents in a table.  The table shows the 'document_id' and the cleaned 'text'.

<aside class="positive">
<b>Tip:</b> The cleaning step is important to ensure that the embeddings are generated from consistent and relevant text, improving the accuracy of subsequent analyses.
</aside>

## Embedding Visualization

Duration: 00:15

In this section, you will learn how to visualize document embeddings and perform semantic searches.

1.  **Accessing Embedding Visualization:** Navigate to the "Embedding Visualization" section using the sidebar.

2.  **Interactive Embedding Atlas:** After processing documents, the application generates an interactive visualization of the document embeddings. This atlas displays each document as a point in a 2D space, where documents with similar meanings are located closer to each other. You can hover over each point to see the document's text.

3.  **Semantic Search:** Use the "Semantic Search" feature to find documents similar to a given query. Enter your search query in the text input field and specify the number of top matches you want to retrieve.  The application uses cosine similarity to compare the query embedding with the embeddings of each document.

    **Cosine Similarity:** The cosine similarity measures the similarity between two vectors (embeddings) by calculating the cosine of the angle between them.  The formula is:

    $$\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} A_i^2}}$$

    where $A$ and $B$ are the embedding vectors.

4.  **Search Results:** The application displays the search results in a table, showing the document ID and its similarity score to the query.

5. **Data Selection**: The interactive `embedding_atlas` component allows you to select data points based on a specified condition. The selected data will be displayed in a table below the atlas.

<aside class="positive">
<b>Tip:</b> Experiment with different search queries to see how the application retrieves relevant documents based on semantic similarity.
</aside>

<aside class="negative">
<b>Warning:</b> Ensure that you have processed documents in the "Document Processing" section before using the "Embedding Visualization" section; otherwise, you'll receive a warning.
</aside>

## Evaluation Metrics

Duration: 00:15

This section guides you on how to evaluate the quality of search results using embedding-based metrics.  These metrics help quantify how relevant, grounded, complete, and answerable a given context is to a query and answer.

1.  **Accessing Evaluation Metrics:** Navigate to the "Evaluation Metrics" section using the sidebar.

2.  **Inputting Text:** Enter a query, a context, and an answer in the provided text areas. These represent:

    *   **Query:** The original search query.
    *   **Context:** The retrieved document or passage.
    *   **Answer:**  An answer generated based on the context.

3.  **Viewing Metrics:**  The application calculates and displays the following metrics:

    *   **Context Relevancy:** Measures how relevant the context is to the query.
    *   **Groundedness:** Measures how well the answer is supported by the context.
    *   **Completeness:** Measures how complete the context is in relation to the answer.
    *   **Answer Relevancy:** Measures how relevant the answer is to the query.

4.  **Understanding Metrics:** Each metric provides insights into the quality of the search results and the coherence between the query, context, and answer. Higher scores generally indicate better quality.

<aside class="positive">
<b>Tip:</b> Use these metrics to fine-tune your search strategies and improve the quality of retrieved information. Experiment with different combinations of queries, contexts, and answers to see how the metrics change.
</aside>
