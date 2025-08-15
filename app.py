"""import streamlit as st
st.set_page_config(page_title=\"QuLab\", layout=\"wide\", initial_sidebar_state=\"expanded\")

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg", width=200)
st.sidebar.title("QuLab: Visual Embedding Lab")
st.sidebar.markdown("Explore semantic relationships in financial documents.")

st.title("QuLab")
st.markdown("""
Welcome to the Visual Embedding Lab! This application allows you to explore semantic relationships within a corpus of financial documents using interactive visualizations and evaluation metrics.

**Learning Objectives:**

*   Understand document embeddings and their semantic meaning.
*   Use `embedding_atlas` for interactive visualization of embeddings.
*   Perform semantic search using cosine similarity.
*   Evaluate the quality of retrieved information using embedding-based metrics.

---

**Navigation**

Use the sidebar to navigate between the different sections of the lab:

*   **Document Processing:** Upload and preprocess financial documents.
*   **Embedding Visualization:** Visualize document embeddings in an interactive atlas.
*   **Evaluation Metrics:** Evaluate the quality of search results.
""")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a section:", (
    "Document Processing",
    "Embedding Visualization",
    "Evaluation Metrics"
))

if page == "Document Processing":
    from application_pages.document_processing import run_document_processing
    run_document_processing()
elif page == "Embedding Visualization":
    from application_pages.embedding_visualization import run_embedding_visualization
    run_embedding_visualization()
elif page == "Evaluation Metrics":
    from application_pages.evaluation_metrics import run_evaluation_metrics

"""
Created by: Gemini
"""
