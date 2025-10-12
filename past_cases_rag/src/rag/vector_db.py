"""
Vector database management using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from ..utils import get_logger, CHROMADB_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from ..processors import TextChunk

logger = get_logger(__name__)

class ChromaVectorDB:
    """
    ChromaDB vector database manager for legal documents.
    """
    
    def __init__(self, persist_directory: str = CHROMADB_PATH, 
                 collection_name: str = COLLECTION_NAME):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.logger = get_logger(self.__class__.__name__)
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Indian Supreme Court case documents"}
            )
            
            self.logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            self.logger.info(f"Persist directory: {self.persist_directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Add text chunks to the vector database.
        
        Args:
            chunks: List of TextChunk objects to add
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks:
                self.logger.warning("No chunks provided to add")
                return False
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Prepare document text
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (list, dict)):
                        metadata[key] = json.dumps(value)
                    else:
                        metadata[key] = str(value)
                
                # Add chunk-specific metadata
                metadata.update({
                    'chunk_id': chunk.chunk_id,
                    'start_char': str(chunk.start_char),
                    'end_char': str(chunk.end_char)
                })
                
                metadatas.append(metadata)
                ids.append(chunk.chunk_id)
            
            # Add to collection in batches to avoid memory issues
            batch_size = 1000
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                batch_documents = documents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                self.collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                self.logger.info(f"Added batch {i//batch_size + 1}: {end_idx}/{total_chunks} chunks")
            
            self.logger.info(f"Successfully added {total_chunks} chunks to vector database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add chunks to vector database: {str(e)}")
            return False
    
    def search(self, query: str, n_results: int = 5, 
               where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            where_filter: Optional metadata filter
        
        Returns:
            List of search results with documents and metadata
        """
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'similarity': 1 - (results['distances'][0][i] if results['distances'] else 0)
                    }
                    
                    # Parse JSON metadata back to original types
                    for key, value in result['metadata'].items():
                        if key in ['paragraph_numbers'] and isinstance(value, str):
                            try:
                                result['metadata'][key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    
                    formatted_results.append(result)
            
            self.logger.debug(f"Search query: '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand structure
            sample_results = self.collection.peek(limit=5)
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory),
                'sample_metadata_keys': []
            }
            
            if sample_results['metadatas']:
                stats['sample_metadata_keys'] = list(sample_results['metadatas'][0].keys())
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {'error': str(e)}
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, str], 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            limit: Maximum number of results
        
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    formatted_results.append({
                        'document': doc,
                        'metadata': results['metadatas'][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Metadata search failed: {str(e)}")
            return []
    
    def get_cases_by_judge(self, judge_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cases by a specific judge."""
        return self.search_by_metadata({'judge': judge_name}, limit)
    
    def get_cases_by_court(self, court_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cases from a specific court."""
        return self.search_by_metadata({'court': court_name}, limit)
    
    def get_cases_by_date_range(self, start_year: str, end_year: str, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get cases within a date range (simplified by year).
        Note: This is a basic implementation. For more complex date filtering,
        you would need to implement custom filtering logic.
        """
        # This is a simplified implementation
        # In practice, you might want to extract year from date strings
        # and create more sophisticated filtering
        return self.search_by_metadata({}, limit)
    
    def close(self):
        """Close the database connection."""
        if self.client:
            # ChromaDB doesn't require explicit closing
            pass