import os
import shutil
import tempfile
import numpy as np
import pytest
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store import FAISSVectorStore

def test_embedding_generation_and_cosine_similarity():
    service = EmbeddingService(api_key="mock_key_for_testing")
    vec1 = service.get_embedding("Python FastApi RAG Assistant")
    vec2 = service.get_embedding("Python FastApi RAG Assistant")
    vec3 = service.get_embedding("Cooking organic pasta recipe")

    assert len(vec1) == 1536
    sim_identical = EmbeddingService.compute_cosine_similarity(vec1, vec2)
    sim_different = EmbeddingService.compute_cosine_similarity(vec1, vec3)

    assert sim_identical > 0.99
    assert sim_identical >= sim_different

def test_faiss_vector_store_add_search_delete():
    temp_dir = tempfile.mkdtemp()
    try:
        store = FAISSVectorStore(vector_dir=temp_dir, dimension=1536)
        service = EmbeddingService(api_key="mock_key_for_testing")

        texts = ["Resume chunk about Python and GenAI", "Project notes on database optimization"]
        embeddings = service.get_embeddings_batch(texts)
        metadatas = [
            {"document_id": 1, "filename": "resume.pdf", "chunk_id": 0, "content": texts[0]},
            {"document_id": 2, "filename": "notes.txt", "chunk_id": 0, "content": texts[1]}
        ]

        store.add_chunks(embeddings, metadatas)
        assert store.index.ntotal == 2

        query_vec = service.get_embedding("Python skills in resume")
        results = store.search(query_vec, top_k=2)

        assert len(results) > 0
        top_meta, score = results[0]
        assert "filename" in top_meta

        # Delete document 1
        store.delete_document_chunks(document_id=1)
        assert store.index.ntotal == 1
        assert store.metadata[0]["document_id"] == 2
    finally:
        shutil.rmtree(temp_dir)
