import pytest
import pandas as pd
from definition_42c2095bf681451fa73a21a4362ac786 import extract_text_from_pdfs
from unittest.mock import patch, mock_open

@pytest.fixture
def mock_pdf_content():
    return "Mock PDF Content"

@patch('pypdf.PdfReader')
def test_extract_text_from_pdfs_single_pdf(mock_pdf_reader, mock_pdf_content):
    mock_pdf_reader.return_value.pages = [mock_pdf_reader.return_value]
    mock_pdf_reader.return_value.extract_text.return_value = mock_pdf_content
    paths = ['test.pdf']
    df = extract_text_from_pdfs(paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df['document_id'][0] == 'test.pdf'
    assert df['text'][0] == mock_pdf_content

@patch('pypdf.PdfReader')
def test_extract_text_from_pdfs_multiple_pdfs(mock_pdf_reader, mock_pdf_content):
    mock_pdf_reader.return_value.pages = [mock_pdf_reader.return_value]
    mock_pdf_reader.return_value.extract_text.return_value = mock_pdf_content
    paths = ['test1.pdf', 'test2.pdf']
    df = extract_text_from_pdfs(paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df['document_id'][0] == 'test1.pdf'
    assert df['document_id'][1] == 'test2.pdf'
    assert df['text'][0] == mock_pdf_content
    assert df['text'][1] == mock_pdf_content

def test_extract_text_from_pdfs_empty_list():
    paths = []
    df = extract_text_from_pdfs(paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

@patch('pypdf.PdfReader', side_effect=FileNotFoundError)
def test_extract_text_from_pdfs_file_not_found(mock_pdf_reader):
    paths = ['nonexistent.pdf']
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdfs(paths)

@patch('pypdf.PdfReader')
def test_extract_text_from_pdfs_no_text_in_pdf(mock_pdf_reader):
    mock_pdf_reader.return_value.pages = [mock_pdf_reader.return_value]
    mock_pdf_reader.return_value.extract_text.return_value = ""
    paths = ['empty.pdf']
    df = extract_text_from_pdfs(paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df['document_id'][0] == 'empty.pdf'
    assert df['text'][0] == ""
