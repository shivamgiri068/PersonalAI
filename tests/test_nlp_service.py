import pytest
from backend.app.services.nlp_service import NLPService

def test_clean_text():
    raw = "  Hello \t world!\r\nThis is   a test. \n\n"
    clean = NLPService.clean_text(raw)
    assert "Hello world!" in clean
    assert "\r" not in clean

def test_character_and_word_count():
    text = "Generative AI Engineering with Python"
    assert NLPService.count_characters(text) == len(text)
    assert NLPService.count_words(text) == 5

def test_token_count():
    text = "PersonalAI is a RAG assistant built for Generative AI jobs."
    token_cnt = NLPService.count_tokens(text)
    assert token_cnt > 0

def test_chunk_text():
    long_text = "Sentence one. " * 30  # ~420 chars
    chunks = NLPService.chunk_text(long_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert "content" in chunks[0]
    assert "chunk_id" in chunks[0]
    assert chunks[0]["chunk_id"] == 0
