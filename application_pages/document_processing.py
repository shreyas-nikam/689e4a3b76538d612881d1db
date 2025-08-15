"""import streamlit as st
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
    text = \" \".join(text.split())
    return text

def run_document_processing():
    st.header(\"Document Processing\")
    st.markdown(\"\"\"In this section, upload financial documents (PDFs) and preprocess them for analysis.\"\"\")

    uploaded_files = st.file_uploader(\"Upload Financial Documents (PDFs)\", type=[\"pdf\"], accept_multiple_files=True)

    if uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files
        # Example usage (replace with your actual file paths)
        #sample_files = [\"Apple_Form_10K.pdf\", \"Tesla_Form_10K.pdf\", \"JPMorgan_Annual_Report.pdf\", \"Amazon_Annual_Report.pdf\"]

    if 'uploaded_files' in st.session_state and st.session_state['uploaded_files']:
        uploaded_files = st.session_state['uploaded_files']
        file_paths = [file.name for file in uploaded_files]

        df = extract_text_from_pdfs(uploaded_files)
        df['text'] = df['text'].apply(clean_text)

        st.subheader(\"Processed Documents\")
        st.dataframe(df)


        if 'processed_df' not in st.session_state:
            st.session_state['processed_df'] = df


"""