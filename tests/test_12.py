import pytest
import pandas as pd
from definition_6ef9c486d9544144befb7fe2749ec0d3 import compute_text_projection

def test_compute_text_projection_empty_df():
    df = pd.DataFrame()
    with pytest.raises(Exception):
        compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")

def test_compute_text_projection_missing_column():
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    with pytest.raises(KeyError):
        compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")

def test_compute_text_projection_valid_df():
    df = pd.DataFrame({"text": ["doc1", "doc2"]})
    compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")
    assert "x" in df.columns
    assert "y" in df.columns
    assert "neighbors" in df.columns
    assert len(df) == 2

def test_compute_text_projection_numeric_text():
    df = pd.DataFrame({"text": [1, 2]})
    with pytest.raises(TypeError):
        compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")

def test_compute_text_projection_nan_text():
    df = pd.DataFrame({"text": [float('NaN'), "doc2"]})
    with pytest.raises(TypeError):
        compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")
