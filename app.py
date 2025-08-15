"""import streamlit as st
st.set_page_config(page_title=\"QuLab\", layout=\"wide\")
st.sidebar.image(\"https://www.quantuniversity.com/assets/img/logo5.jpg\")
st.sidebar.divider()
st.title(\"QuLab\")
st.divider()
st.markdown("""
In this lab, we explore visual embeddings of financial documents. The application allows you to upload documents, visualize their embeddings in an interactive atlas, perform semantic searches, and evaluate retrieval quality using embedding-based metrics.

**Learning Goals:**

*   Understand how embeddings encode document meaning.
*   Use `embedding_atlas.projection.compute_text_projection` to create 2D coordinates for visualization.
*   Display interactive visualizations with `embedding_atlas.streamlit.embedding_atlas`.
*   Perform semantic search via cosine similarity.
*   Apply various embedding-based evaluation metrics: Context Relevancy, Groundedness, Completeness, and Answer Relevancy.

### Formulae

**Cosine Similarity**

$$\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
""")
# Your code starts here
page = st.sidebar.selectbox(label=\"Navigation\", options=[\"Document Processing\", \"Embedding Visualization\", \"Evaluation Metrics\"])
if page == \"Document Processing\":
    from application_pages.document_processing import run_document_processing
    run_document_processing()
elif page == \"Embedding Visualization\":
    from application_pages.embedding_visualization import run_embedding_visualization
    run_embedding_visualization()
elif page == \"Evaluation Metrics\":
    from application_pages.evaluation_metrics import run_evaluation_metrics
    run_evaluation_metrics()
# Your code ends
"""