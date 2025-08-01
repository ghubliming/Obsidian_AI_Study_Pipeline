"""
Semantic retrieval module using sentence transformers and FAISS for efficient similarity search.
"""

import os
import pickle
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..preprocessing import ContentChunk

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """Semantic retrieval system for finding relevant content chunks."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 cache_dir: Optional[str] = None):
        """
        Initialize semantic retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            cache_dir: Directory to cache embeddings and index
        """
        self.model_name = model_name
        self.index_type = index_type
        self.cache_dir = cache_dir
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        logger.info(f"Initialized SemanticRetriever with {model_name}")
    
    def build_index(self, chunks: List[ContentChunk], force_rebuild: bool = False) -> None:
        """
        Build semantic search index from content chunks.
        
        Args:
            chunks: List of ContentChunk objects
            force_rebuild: Whether to force rebuilding even if cache exists
        """
        self.chunks = chunks
        
        # Check for cached embeddings
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, f"embeddings_{self.model_name.replace('/', '_')}.pkl")
        
        # Load or compute embeddings
        if cache_path and os.path.exists(cache_path) and not force_rebuild:
            logger.info("Loading cached embeddings...")
            with open(cache_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            logger.info(f"Computing embeddings for {len(chunks)} chunks...")
            texts = [chunk.text for chunk in chunks]
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Cache embeddings
            if cache_path:
                logger.info(f"Caching embeddings to {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.embeddings, f)
        
        # Build FAISS index
        self._build_faiss_index()
        
        logger.info(f"Built semantic index with {len(chunks)} chunks")
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index from embeddings."""
        if self.embeddings is None:
            raise ValueError("Embeddings not computed. Call build_index first.")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        if self.index_type == "flat":
            # Flat index for exact search
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, len(self.chunks) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings_normalized.astype(np.float32))
        else:
            # Default to flat index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings_normalized.astype(np.float32))
        
        logger.info(f"Built {self.index_type} FAISS index with {self.index.ntotal} vectors")
    
    def search(self, 
               query: str, 
               k: int = 5,
               filter_by_type: Optional[str] = None,
               filter_by_tags: Optional[List[str]] = None) -> List[Tuple[ContentChunk, float]]:
        """
        Search for relevant content chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_by_type: Filter results by chunk type
            filter_by_tags: Filter results by tags
            
        Returns:
            List of (ContentChunk, similarity_score) tuples
        """
        if self.index is None or not self.chunks:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search index
        search_k = min(k * 3, len(self.chunks))  # Search more to allow for filtering
        similarities, indices = self.index.search(query_normalized.astype(np.float32), search_k)
        
        # Collect results with filtering
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            
            # Apply filters
            if filter_by_type and chunk.chunk_type != filter_by_type:
                continue
            
            if filter_by_tags and chunk.metadata:
                chunk_tags = chunk.metadata.get('tags', [])
                if not any(tag in chunk_tags for tag in filter_by_tags):
                    continue
            
            results.append((chunk, float(similarity)))
            
            if len(results) >= k:
                break
        
        logger.debug(f"Search query '{query}' returned {len(results)} results")
        return results
    
    def search_by_topic(self, 
                       topic: str,
                       k: int = 10,
                       diversity_threshold: float = 0.8) -> List[Tuple[ContentChunk, float]]:
        """
        Search for content chunks on a specific topic with diversity.
        
        Args:
            topic: Topic to search for
            k: Number of results to return
            diversity_threshold: Similarity threshold for diversity filtering
            
        Returns:
            List of diverse content chunks on the topic
        """
        # Initial search with higher k
        initial_results = self.search(topic, k=k*2)
        
        if not initial_results:
            return []
        
        # Diversify results to avoid too similar chunks
        diverse_results = [initial_results[0]]  # Always include top result
        
        for chunk, score in initial_results[1:]:
            # Check if this chunk is too similar to already selected chunks
            is_diverse = True
            for selected_chunk, _ in diverse_results:
                chunk_embedding = self.model.encode([chunk.text])
                selected_embedding = self.model.encode([selected_chunk.text])
                
                # Compute cosine similarity
                similarity = np.dot(chunk_embedding[0], selected_embedding[0]) / (
                    np.linalg.norm(chunk_embedding[0]) * np.linalg.norm(selected_embedding[0])
                )
                
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append((chunk, score))
            
            if len(diverse_results) >= k:
                break
        
        return diverse_results
    
    def get_related_chunks(self, 
                          chunk: ContentChunk,
                          k: int = 5,
                          same_note_only: bool = False) -> List[Tuple[ContentChunk, float]]:
        """
        Find chunks related to a given chunk.
        
        Args:
            chunk: Reference chunk
            k: Number of related chunks to return
            same_note_only: Whether to only return chunks from the same note
            
        Returns:
            List of related chunks with similarity scores
        """
        results = self.search(chunk.text, k=k+1)  # +1 because the chunk itself might be included
        
        # Filter out the original chunk and apply same_note filter
        filtered_results = []
        for related_chunk, score in results:
            if related_chunk.text == chunk.text:  # Skip the original chunk
                continue
            
            if same_note_only and related_chunk.source_note != chunk.source_note:
                continue
            
            filtered_results.append((related_chunk, score))
            
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save chunks and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'model_name': self.model_name,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved index to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Load the FAISS index and chunks from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load chunks and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            # Verify model compatibility
            if data['model_name'] != self.model_name:
                logger.warning(f"Loaded index was built with {data['model_name']}, "
                             f"but current model is {self.model_name}")
        
        logger.info(f"Loaded index from {filepath}")
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if not self.chunks:
            return {"status": "empty"}
        
        chunk_types = {}
        notes = set()
        total_tags = set()
        
        for chunk in self.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            notes.add(chunk.source_note)
            if chunk.metadata and 'tags' in chunk.metadata:
                total_tags.update(chunk.metadata['tags'])
        
        return {
            "total_chunks": len(self.chunks),
            "unique_notes": len(notes),
            "unique_tags": len(total_tags),
            "chunk_types": chunk_types,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_type,
            "model_name": self.model_name,
            "index_size": self.index.ntotal if self.index else 0
        }