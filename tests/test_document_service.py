import os
import tempfile
import pytest
from backend.app.services.document_service import DocumentService

def test_extract_text_file():
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write("Sample document content for testing document extraction.")
        tmp_path = tmp.name

    try:
        extracted = DocumentService.extract_text_from_file(tmp_path, "txt")
        assert "Sample document content" in extracted
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_pandas_document_analytics():
    docs = [
        {"id": 1, "filename": "resume.pdf", "file_type": "PDF", "chunk_count": 5, "char_count": 2500, "word_count": 400, "token_count": 520},
        {"id": 2, "filename": "notes.txt", "file_type": "TXT", "chunk_count": 3, "char_count": 1200, "word_count": 200, "token_count": 260}
    ]
    stats = DocumentService.generate_pandas_document_analytics(docs)
    assert stats["total_documents"] == 2
    assert stats["total_chunks"] == 8
    assert stats["total_characters"] == 3700
    assert stats["total_words"] == 600
    assert stats["total_tokens"] == 780
    assert stats["average_chunk_length"] == 462.5
    assert stats["file_type_distribution"]["PDF"] == 1
    assert stats["file_type_distribution"]["TXT"] == 1
