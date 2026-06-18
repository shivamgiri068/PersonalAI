import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from backend.app.config import settings

class FAISSVectorStore:
    def __init__(self, vector_dir: str = None, dimension: int = 1536):
        self.vector_dir = vector_dir or settings.VECTOR_STORE_DIR
        self.dimension = dimension
        self.index_path = os.path.join(self.vector_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.vector_dir, "metadata.pkl")
        
        os.makedirs(self.vector_dir, exist_ok=True)
        
        self.metadata: List[Dict[str, Any]] = []
        self.index = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        """
        Loads existing FAISS index & metadata, or creates a new IndexFlatIP (Inner Product / Cosine Similarity).
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                return
            except Exception:
                pass
        
        # Initialize L2 or Inner Product index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def save(self):
        """
        Persists FAISS index binary and metadata array to disk.
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

    def add_chunks(self, embeddings: List[List[float]], chunk_metadatas: List[Dict[str, Any]]):
        """
        Adds vectors and corresponding metadata to FAISS index.
        """
        if not embeddings or not chunk_metadatas:
            return
        
        vectors = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity (Inner Product on L2 normalized vectors = Cosine Similarity)
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        self.metadata.extend(chunk_metadatas)
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches FAISS index for top_k most similar vectors.
        Returns list of tuples (metadata_dict, similarity_score).
        """
        if self.index is None or self.index.ntotal == 0 or not self.metadata:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
                
        return results

    def delete_document_chunks(self, document_id: int):
        """
        Removes all chunks associated with document_id and rebuilds the FAISS index.
        """
        if not self.metadata or self.index is None or self.index.ntotal == 0:
            return

        # Filter out metadata belonging to document_id
        keep_indices = [i for i, meta in enumerate(self.metadata) if meta.get("document_id") != document_id]
        
        if len(keep_indices) == len(self.metadata):
            return  # Nothing to delete

        if not keep_indices:
            # All items deleted
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.save()
            return

        # Reconstruct vector array for remaining items
        remaining_vectors = []
        for i in keep_indices:
            vec = np.zeros(self.dimension, dtype=np.float32)
            self.index.reconstruct(i, vec)
            remaining_vectors.append(vec)

        # Create new index
        new_index = faiss.IndexFlatIP(self.dimension)
        if remaining_vectors:
            vectors_array = np.array(remaining_vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_array)
            new_index.add(vectors_array)
            
        self.index = new_index
        self.metadata = [self.metadata[i] for i in keep_indices]
        self.save()
