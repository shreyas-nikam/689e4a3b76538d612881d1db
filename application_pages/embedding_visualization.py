"""import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from atlas_embed import AtlasEmbed
import duckdb

def load_sentence_bert_model(model_name=\"all-MiniLM-L6-v2\"):
    """Loads a Sentence-BERT model."""
    model = SentenceTransformer(model_name)
    return model

def generate_embeddings(df, model):
    """Generates dense vector embeddings for each document in the DataFrame."""
    df['embeddings'] = df['text'].apply(lambda x: model.encode(str(x)) if x is not None else model.encode(""))
    return df

def compute_text_projection(df, text, x, y, neighbors):
    """Computes a 2D projection of text embeddings for visualization."""
    atlas = AtlasEmbed(df=df, text=text)
    df = atlas.embed(x=x, y=y, neighbors=neighbors)
    return df

def load_data_from_duckdb(df, predicate):
    """Loads data from DuckDB based on predicate.
    Args:
        df (pd.DataFrame): Input DataFrame.
        predicate (str): DuckDB predicate.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if predicate is None:
        raise TypeError(\"Predicate cannot be None\")

    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        con.register('df', df)
        selection = con.execute(f\"SELECT * FROM df WHERE {predicate}\"").fetchdf()
        con.close()
        return selection
    except Exception as e:
        raise NameError(f\"Error executing predicate: {e}\")


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

def compute_cosine_similarity(embedding1, embedding2):
    """Computes cosine similarity between two embeddings."""
    import numpy as np
    norm_1 = np.linalg.norm(embedding1)
    norm_2 = np.linalg.norm(embedding2)
    similarity = np.dot(embedding1, embedding2) / (norm_1 * norm_2)
    return similarity

def run_embedding_visualization():
    st.header(\"Embedding Visualization\")
    st.markdown(\"\"\"In this section, visualize the document embeddings in an interactive atlas and perform semantic searches.\"\"\")

    if 'processed_df' not in st.session_state:
        st.warning(\"Please upload and process documents in the Document Processing section first.\")
        return

    df = st.session_state['processed_df'].copy()

    # Embedding Generation
    if 'embeddings' not in df.columns:
        model = load_sentence_bert_model()
        df = generate_embeddings(df, model)
        st.session_state['processed_df'] = df # Update the session state

    # Projection for Visualization
    if 'projection_x' not in df.columns:
        df = compute_text_projection(df, text='text', x='projection_x', y='projection_y', neighbors='neighbors')
        st.session_state['processed_df'] = df # Update the session state

    # Interactive Visualization
    st.subheader(\"Interactive Embedding Atlas\")
    value = embedding_atlas(df, text='text', x='projection_x', y='projection_y', neighbors='neighbors', show_table=True)

    predicate = value.get(\"predicate\")
    if predicate is not None:
        try:
            selection = load_data_from_duckdb(df, predicate)
            st.subheader(\"Selected Data\")
            st.dataframe(selection)
        except NameError as e:
            st.error(f\"Error loading data from DuckDB: {e}\")

    # Semantic Search
    st.subheader(\"Semantic Search\")
    search_query = st.text_input(\"Enter your search query:\")
    num_results = st.number_input(\"Number of top matches to return:\", min_value=1, value=5)

    if search_query:
        model = load_sentence_bert_model()
        query_embedding = model.encode(search_query)
        similarities = semantic_similarity_search(df, query_embedding, num_results)

        st.subheader(\"Search Results\")
        search_results_df = pd.DataFrame(similarities, columns=['document_id', 'similarity_score'])
        st.dataframe(search_results_df)


"""