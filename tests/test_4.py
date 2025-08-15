import pytest
import pandas as pd
from unittest.mock import patch
from definition_d7933933b5ec4ce5a0f487ebf895ea89 import compute_text_projection


def test_compute_text_projection_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(Exception):
        compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")


def test_compute_text_projection_missing_text_column():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(KeyError):
        compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")


def test_compute_text_projection_valid_dataframe():
    df = pd.DataFrame({"text": ["This is document 1", "This is document 2"]})
    df = compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")
    assert "x" in df.columns
    assert "y" in df.columns
    assert "neighbors" in df.columns
    assert len(df) == 2



def test_compute_text_projection_numerical_text_column():
    df = pd.DataFrame({"text": [1, 2]})
    df = compute_text_projection(df, text="text", x="x", y="y", neighbors="neighbors")
    assert "x" in df.columns
    assert "y" in df.columns
    assert "neighbors" in df.columns
    assert len(df) == 2


def test_compute_text_projection_different_column_names():
    df = pd.DataFrame({"my_text": ["This is document 1", "This is document 2"]})
    df = compute_text_projection(df, text="my_text", x="proj_x", y="proj_y", neighbors="nearby")
    assert "proj_x" in df.columns
    assert "proj_y" in df.columns
    assert "nearby" in df.columns
    assert len(df) == 2