import os
import json
import numpy as np
import os
from openai import OpenAI
from pypdf import PdfReader

INDEX_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

class VectorStore:
    def __init__(self, index_dir=INDEX_DIR):
        self.index_dir = index_dir
        self.embeddings_path = os.path.join(index_dir, "embeddings.npy")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        self.embeddings = None
        self.metadata = []
        self.load_index()

    def is_indexed(self):
        return (
            os.path.exists(self.embeddings_path) and 
            os.path.exists(self.metadata_path)
        )

    def load_index(self):
        if not self.is_indexed():
            print(f"Warning: Index files not found in {self.index_dir}. Run indexer.py first.")
            return False

        self.embeddings = np.load(self.embeddings_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Pre-normalize embeddings for fast cosine similarity calculation
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        self.normalized_embeddings = self.embeddings / norms
        print(f"Loaded vector store with {len(self.metadata)} chunks.")
        return True

    def similarity_search(self, query_embedding, top_k=5):
        """
        Performs vector similarity search against normalized stored embeddings.
        Returns top_k matching chunks with similarity scores.
        """
        if self.embeddings is None or len(self.metadata) == 0:
            raise ValueError("VectorStore is empty or not loaded.")

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity is dot product of normalized vectors
        similarities = np.dot(self.normalized_embeddings, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_info = dict(self.metadata[idx])
            chunk_info["similarity_score"] = float(similarities[idx])
            results.append(chunk_info)

        return results

if __name__ == "__main__":
    store = VectorStore()
    if store.is_indexed():
        print("Vector store initialized successfully.")
    else:
        print("Vector store initialization failed.")
