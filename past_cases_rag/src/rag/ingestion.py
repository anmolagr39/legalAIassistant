"""
Document ingestion pipeline for processing and storing legal documents.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import os
from tqdm import tqdm

from ..utils import get_logger, CASE_DOCS_PATH
from ..processors import LegalDocumentProcessor, LegalTextChunker, ProcessedDocument, TextChunk
from .vector_db import ChromaVectorDB

logger = get_logger(__name__)

class DocumentIngestionPipeline:
    """
    Complete pipeline for processing legal documents and storing in vector database.
    """
    
    def __init__(self, docs_path: str = CASE_DOCS_PATH, 
                 vector_db: Optional[ChromaVectorDB] = None):
        self.docs_path = Path(docs_path)
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.doc_processor = LegalDocumentProcessor()
        self.text_chunker = LegalTextChunker()
        self.vector_db = vector_db or ChromaVectorDB()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_docs': 0,
            'failed_docs': 0,
            'total_chunks': 0,
            'ingested_chunks': 0
        }
    
    def get_document_paths(self) -> List[str]:
        """
        Get all document file paths from the documents directory.
        
        Returns:
            List of document file paths
        """
        if not self.docs_path.exists():
            self.logger.error(f"Documents directory not found: {self.docs_path}")
            return []
        
        # Get all .txt files
        doc_paths = list(self.docs_path.glob("*.txt"))
        doc_paths = [str(path) for path in doc_paths]
        
        self.logger.info(f"Found {len(doc_paths)} document files")
        self.stats['total_files'] = len(doc_paths)
        
        return doc_paths
    
    def process_documents(self, doc_paths: List[str]) -> List[ProcessedDocument]:
        """
        Process all documents using the document processor.
        
        Args:
            doc_paths: List of document file paths
        
        Returns:
            List of ProcessedDocument objects
        """
        self.logger.info("Starting document processing...")
        
        processed_docs = []
        
        for doc_path in tqdm(doc_paths, desc="Processing documents"):
            processed_doc = self.doc_processor.process_document(doc_path)
            if processed_doc:
                processed_docs.append(processed_doc)
                self.stats['processed_docs'] += 1
            else:
                self.stats['failed_docs'] += 1
        
        self.logger.info(f"Document processing completed: {len(processed_docs)} successful, "
                        f"{self.stats['failed_docs']} failed")
        
        return processed_docs
    
    def create_chunks(self, processed_docs: List[ProcessedDocument]) -> List[TextChunk]:
        """
        Create text chunks from processed documents.
        
        Args:
            processed_docs: List of ProcessedDocument objects
        
        Returns:
            List of TextChunk objects
        """
        self.logger.info("Starting text chunking...")
        
        all_chunks = []
        
        for processed_doc in tqdm(processed_docs, desc="Creating chunks"):
            chunks = self.text_chunker.chunk_document(processed_doc)
            all_chunks.extend(chunks)
        
        self.stats['total_chunks'] = len(all_chunks)
        
        self.logger.info(f"Text chunking completed: {len(all_chunks)} chunks created")
        
        return all_chunks
    
    def ingest_to_vector_db(self, chunks: List[TextChunk]) -> bool:
        """
        Ingest chunks into the vector database.
        
        Args:
            chunks: List of TextChunk objects
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Starting vector database ingestion...")
        
        success = self.vector_db.add_chunks(chunks)
        
        if success:
            self.stats['ingested_chunks'] = len(chunks)
            self.logger.info(f"Vector database ingestion completed: {len(chunks)} chunks ingested")
        else:
            self.logger.error("Vector database ingestion failed")
        
        return success
    
    def run_full_pipeline(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete document ingestion pipeline.
        
        Args:
            limit: Optional limit on number of documents to process
        
        Returns:
            Dictionary with pipeline statistics
        """
        self.logger.info("Starting full document ingestion pipeline...")
        
        try:
            # Step 1: Get document paths
            doc_paths = self.get_document_paths()
            if not doc_paths:
                raise ValueError("No documents found to process")
            
            # Limit documents if specified
            if limit:
                doc_paths = doc_paths[:limit]
                self.logger.info(f"Limited processing to {limit} documents")
            
            # Step 2: Process documents
            processed_docs = self.process_documents(doc_paths)
            if not processed_docs:
                raise ValueError("No documents were successfully processed")
            
            # Step 3: Create chunks
            chunks = self.create_chunks(processed_docs)
            if not chunks:
                raise ValueError("No chunks were created")
            
            # Step 4: Ingest into vector database
            success = self.ingest_to_vector_db(chunks)
            if not success:
                raise ValueError("Vector database ingestion failed")
            
            # Step 5: Save metadata summary
            summary_path = "ingestion_summary.json"
            self.doc_processor.save_metadata_summary(processed_docs, summary_path)
            
            self.logger.info("Document ingestion pipeline completed successfully!")
            
            return {
                'status': 'success',
                'statistics': self.stats,
                'summary_file': summary_path
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'statistics': self.stats
            }
    
    def run_incremental_ingestion(self, doc_paths: List[str]) -> Dict[str, Any]:
        """
        Run incremental ingestion for specific documents.
        
        Args:
            doc_paths: List of specific document paths to process
        
        Returns:
            Dictionary with ingestion results
        """
        self.logger.info(f"Starting incremental ingestion for {len(doc_paths)} documents...")
        
        try:
            # Process and ingest documents
            processed_docs = self.process_documents(doc_paths)
            chunks = self.create_chunks(processed_docs)
            success = self.ingest_to_vector_db(chunks)
            
            if success:
                return {
                    'status': 'success',
                    'processed_docs': len(processed_docs),
                    'created_chunks': len(chunks)
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'Vector database ingestion failed'
                }
                
        except Exception as e:
            self.logger.error(f"Incremental ingestion failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        # Add vector database stats
        db_stats = self.vector_db.get_collection_stats()
        
        return {
            'pipeline_stats': self.stats,
            'vector_db_stats': db_stats
        }