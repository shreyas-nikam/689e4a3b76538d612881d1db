import pytest
import pandas as pd
from definition_50faf3322afe4c7bb5915a9c322ce37b import load_documents_into_dataframe

@pytest.fixture
def mock_pdf_files(tmp_path):
    # Create mock PDF files for testing
    file1 = tmp_path / "test_doc1.pdf"
    file1.write_bytes(b"%PDF-1.0\nDummy content for test_doc1.pdf")  # Minimal valid PDF content
    file2 = tmp_path / "test_doc2.pdf"
    file2.write_bytes(b"%PDF-1.0\nDummy content for test_doc2.pdf")
    return [str(file1), str(file2)]

def test_load_documents_into_dataframe_valid_files(mock_pdf_files):
    df = load_documents_into_dataframe(mock_pdf_files)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "document_id" in df.columns
    assert "title" in df.columns
    assert "text" in df.columns
    assert df["title"][0] == "test_doc1.pdf"
    assert df["title"][1] == "test_doc2.pdf"
    assert "Dummy content" in df["text"][0]

def test_load_documents_into_dataframe_empty_file_list():
    df = load_documents_into_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_load_documents_into_dataframe_non_pdf_file(tmp_path):
    file = tmp_path / "test_doc.txt"
    file.write_text("This is not a PDF")
    with pytest.raises(Exception):  # Expecting some kind of exception during PDF parsing
        load_documents_into_dataframe([str(file)])

def test_load_documents_into_dataframe_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_documents_into_dataframe(["nonexistent_file.pdf"])
