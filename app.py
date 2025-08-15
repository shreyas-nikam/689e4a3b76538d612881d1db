import streamlit as st
import pandas as pd
from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer
from embedding_atlas.streamlit import embedding_atlas
from embedding_atlas.projection import compute_text_projection

import duckdb
from pathlib import Path

# -------- Settings --------
SAMPLE_CSV_PATH = Path("data/embedded_documents.csv")  # <-- change if needed

st.set_page_config(page_title="QuLab", layout="wide",
                   initial_sidebar_state="expanded")

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg", width=200)
st.sidebar.markdown("Explore semantic relationships in financial documents.")

st.title("QuLab: Visualize Financial Documents")
st.markdown("""
---
Welcome to the Visual Embedding Lab! This application allows you to explore semantic relationships within a corpus of financial documents using interactive visualizations.

**Learning Objectives:**
* Understand document embeddings and their semantic meaning.
* Visualize document embeddings in an interactive atlas.
* Filter and explore document embeddings based on semantic relationships.
""")

# ---------- Helpers ----------
def extract_text_from_pdfs(paths):
    """Extracts text from PDF documents and stores them in a DataFrame."""
    data = []
    for file in paths:
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            data.append({'document_id': file.name, 'title': None, 'text': text})
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file.name}")
    df = pd.DataFrame(data)
    return df

def split_text_into_sentences(df):
    """Splits text in the DataFrame into sentences, creating a new row for each sentence."""
    if 'text' not in df.columns:
        raise KeyError("The DataFrame must contain a 'text' column.")

    sentence_data = []
    for _, row in df.iterrows():
        text = row['text'] or ""
        document_id = row['document_id']
        # Simple sentence splitting based on common punctuation
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        for sentence in sentences:
            if sentence and sentence.strip():
                sentence_data.append({'document_id': document_id, 'text': sentence.strip()})
    sentence_df = pd.DataFrame(sentence_data)
    return sentence_df

def clean_text(text):
    """Cleans text by lowercasing and removing extra spaces."""
    text = (text or "").lower()
    text = " ".join(text.split())
    return text

@st.cache_data(show_spinner=False)
def load_sample_csv(sample_path: Path) -> pd.DataFrame:
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample CSV not found at: {sample_path.resolve()}")
    df = pd.read_csv(sample_path)
    # Expect at least 'document_id' and 'text'
    required = {'document_id', 'text'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Sample CSV must contain columns: {required}. Found: {set(df.columns)}")
    # Clean (optional) and filter tiny rows
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.len() >= 10].reset_index(drop=True)
    return df

def load_sentence_bert_model(model_name="all-MiniLM-L6-v2"):
    """Loads a Sentence-BERT model."""
    model = SentenceTransformer(model_name)
    return model

def _compute_text_projection(df, text, x, y, neighbors):
    """Computes a 2D projection of text embeddings for visualization."""
    compute_text_projection(df, text=text, x=x, y=y, neighbors=neighbors)

# ---------- UI: Document Processing ----------
def run_document_processing():
    st.header("Document Processing")
    st.markdown("Upload financial PDFs **or** click **Load sample data** to use a pre-saved CSV.")

    # Data source choice
    source = st.radio(
        "Choose data source:",
        ["Upload PDFs", "Load sample dataset (CSV)"],
        horizontal=True
    )

    if source == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload Financial Documents (PDFs)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.session_state['uploaded_files'] = uploaded_files

        if 'uploaded_files' in st.session_state and st.session_state['uploaded_files']:
            uploaded_files = st.session_state['uploaded_files']

            df_docs = extract_text_from_pdfs(uploaded_files)
            df_docs['text'] = df_docs['text'].apply(clean_text)
            df_sent = split_text_into_sentences(df_docs)
            df_sent = df_sent[df_sent['text'].str.len() >= 10].reset_index(drop=True)

            st.subheader("Processed Documents")
            st.dataframe(df_sent, use_container_width=True)

            st.session_state['processed_df'] = df_sent

    else:
        # Load sample dataset via button
        cols = st.columns([1, 2])
        with cols[0]:
            if st.button("📥 Load sample data"):
                try:
                    df_sample = load_sample_csv(SAMPLE_CSV_PATH)
                    st.session_state['processed_df'] = df_sample
                    st.success(f"Loaded sample dataset from: {SAMPLE_CSV_PATH}")
                except Exception as e:
                    st.error(f"Failed to load sample dataset: {e}")

        # Preview if already loaded
        if 'processed_df' in st.session_state:
            st.subheader("Sample Dataset (Preview)")
            st.dataframe(st.session_state['processed_df'].head(100), use_container_width=True)

run_document_processing()

# ---------- UI: Embedding Visualization ----------
def run_embedding_visualization():
    st.header("Embedding Visualization")
    st.markdown("Visualize embeddings in an interactive atlas and filter selections.")

    if 'processed_df' not in st.session_state:
        st.warning("Please upload/process documents or load the sample dataset first.")
        return

    df = st.session_state['processed_df'].copy()

    # Projection for Visualization (Embedding Atlas computes embeddings + projection internally)
    if 'projection_x' not in df.columns or 'projection_y' not in df.columns or 'neighbors' not in df.columns:
        _compute_text_projection(df=df, text='text', x='projection_x', y='projection_y', neighbors='neighbors')
        st.session_state['processed_df'] = df  # Update the session state

    # Interactive Visualization
    st.subheader("Interactive Embedding Atlas")
    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        value = embedding_atlas(
            df,
            text='text',
            x='projection_x',
            y='projection_y',
            neighbors='neighbors',
            show_table=True
        )


run_embedding_visualization()

# ---------- License ----------
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025
This notebook was created for **educational purposes only** and is **not intended for commercial use**.

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.
- You **may not delete or modify this license cell** without authorization.
- This notebook was generated using **QuCreate**, an AI-powered assistant.
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
