import pytest
import pandas as pd
from definition_719353188667420ca806a27283c13848 import load_documents_to_dataframe

@pytest.fixture
def mock_pdf_content(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.pdf"
    p.write_text("This is a test document.\nIt has multiple lines.")
    return str(p)

def test_load_documents_to_dataframe_single_file(mock_pdf_content, monkeypatch):
    # Mock pdf extraction function
    def mock_extract_text_from_pdf(file_path):
        return "This is a test document.\nIt has multiple lines."
    monkeypatch.setattr('definition_719353188667420ca806a27283c13848', 'extract_text_from_pdf', mock_extract_text_from_pdf)

    file_paths = [mock_pdf_content]
    df = load_documents_to_dataframe(file_paths)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert 'document_id' in df.columns
    assert 'title' in df.columns
    assert 'text' in df.columns
    assert df['text'][0] == "This is a test document.\nIt has multiple lines."
    assert df['title'][0] == "test.pdf"

def test_load_documents_to_dataframe_empty_list():
    file_paths = []
    df = load_documents_to_dataframe(file_paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_load_documents_to_dataframe_multiple_files(tmp_path, monkeypatch):
    # Create two mock pdf files
    d = tmp_path / "sub"
    d.mkdir()
    p1 = d / "test1.pdf"
    p1.write_text("This is the first document.")
    p2 = d / "test2.pdf"
    p2.write_text("This is the second document.")

    def mock_extract_text_from_pdf(file_path):
        if "test1.pdf" in file_path:
            return "This is the first document."
        elif "test2.pdf" in file_path:
            return "This is the second document."
        else:
            return ""
    monkeypatch.setattr('definition_719353188667420ca806a27283c13848', 'extract_text_from_pdf', mock_extract_text_from_pdf)
    file_paths = [str(p1), str(p2)]
    df = load_documents_to_dataframe(file_paths)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df['text'][0] == "This is the first document."
    assert df['text'][1] == "This is the second document."
    assert df['title'][0] == "test1.pdf"
    assert df['title'][1] == "test2.pdf"

def test_load_documents_to_dataframe_file_not_found(monkeypatch):
    # Mock pdf extraction function to raise an exception
    def mock_extract_text_from_pdf(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    monkeypatch.setattr('definition_719353188667420ca806a27283c13848', 'extract_text_from_pdf', mock_extract_text_from_pdf)
    file_paths = ["nonexistent_file.pdf"]
    with pytest.raises(FileNotFoundError):
        load_documents_to_dataframe(file_paths)
