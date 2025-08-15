import pytest
from definition_e24f7fb6ff934fedb378cd3270fdb536 import extract_text_from_pdf

@pytest.mark.parametrize("file_path, expected_result", [
    ("non_existent_file.pdf", ""),
    ("empty_file.pdf", ""),
    ("test.pdf", "Extracted text from PDF file.")
])
def test_extract_text_from_pdf(file_path, expected_result, monkeypatch):
    def mock_extract_text(path):
        if path == "test.pdf":
            return "Extracted text from PDF file."
        elif path == "empty_file.pdf":
            return ""
        else:
            raise FileNotFoundError

    monkeypatch.setattr("pypdf.PdfReader", lambda x: mock_extract_text(x))

    try:
        assert extract_text_from_pdf(file_path) == expected_result
    except FileNotFoundError:
        assert file_path == "non_existent_file.pdf"

