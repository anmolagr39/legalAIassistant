"""
Text chunking strategies for legal documents.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

from ..utils import get_logger, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

@dataclass
class TextChunk:
    """Structure for text chunks with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_char: int
    end_char: int
    
class LegalTextChunker:
    """
    Specialized text chunker for legal documents.
    Preserves legal structure while creating semantic chunks.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = get_logger(self.__class__.__name__)
        
        # Legal section patterns
        self.section_patterns = [
            r'^\d+\.\s+',  # Numbered paragraphs
            r'^FACTS?\s*:?',
            r'^ISSUE[S]?\s*:?',
            r'^HELD\s*:?',
            r'^REASONING\s*:?',
            r'^JUDGMENT\s*:?',
            r'^ORDER[S]?\s*:?',
            r'^CONCLUSION\s*:?'
        ]
    
    def identify_legal_boundaries(self, text: str) -> List[int]:
        """
        Identify natural break points in legal text.
        
        Args:
            text: Legal document text
        
        Returns:
            List of character positions for natural breaks
        """
        boundaries = [0]
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for legal section headers
            for pattern in self.section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    if current_pos > 0:
                        boundaries.append(current_pos)
                    break
            
            # Check for paragraph breaks (double newlines in original text)
            if not line_stripped and current_pos > 0:
                if current_pos not in boundaries:
                    boundaries.append(current_pos)
            
            current_pos += len(line) + 1  # +1 for newline character
        
        # Add end position
        boundaries.append(len(text))
        
        return sorted(list(set(boundaries)))
    
    def create_semantic_chunks(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Create semantic chunks preserving legal document structure.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata to include in chunks
        
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # First, identify natural legal boundaries
        boundaries = self.identify_legal_boundaries(text)
        
        # Create initial segments based on boundaries
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment_text = text[start:end].strip()
            
            if segment_text:
                segments.append({
                    'text': segment_text,
                    'start': start,
                    'end': end
                })
        
        # Process segments to create appropriately sized chunks
        current_chunk_text = ""
        current_chunk_start = 0
        chunk_counter = 0
        
        for segment in segments:
            segment_text = segment['text']
            
            # If adding this segment would exceed chunk size
            if len(current_chunk_text) + len(segment_text) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk_text.strip():
                    chunk = self._create_chunk(
                        content=current_chunk_text.strip(),
                        metadata=metadata,
                        chunk_id=f"{metadata.get('case_id', 'unknown')}_{chunk_counter}",
                        start_char=current_chunk_start,
                        end_char=current_chunk_start + len(current_chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                
                # Start new chunk
                # Add overlap from previous chunk if available
                overlap_text = ""
                if current_chunk_text and self.overlap > 0:
                    words = current_chunk_text.split()
                    if len(words) > 10:  # Only add overlap if previous chunk was substantial
                        overlap_words = words[-min(20, len(words)):]  # Last 20 words as overlap
                        overlap_text = " ".join(overlap_words) + " "
                
                current_chunk_text = overlap_text + segment_text
                current_chunk_start = segment['start']
            
            else:
                # Add segment to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + segment_text
                else:
                    current_chunk_text = segment_text
                    current_chunk_start = segment['start']
        
        # Add final chunk if it has content
        if current_chunk_text.strip():
            chunk = self._create_chunk(
                content=current_chunk_text.strip(),
                metadata=metadata,
                chunk_id=f"{metadata.get('case_id', 'unknown')}_{chunk_counter}",
                start_char=current_chunk_start,
                end_char=current_chunk_start + len(current_chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any], 
                     chunk_id: str, start_char: int, end_char: int) -> TextChunk:
        """
        Create a TextChunk object with enhanced metadata.
        
        Args:
            content: Chunk content
            metadata: Original document metadata
            chunk_id: Unique chunk identifier
            start_char: Start character position
            end_char: End character position
        
        Returns:
            TextChunk object
        """
        # Enhance metadata for this chunk
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_size': len(content),
            'word_count': len(content.split()),
            'contains_citation': bool(re.search(r'AIR\s+\d{4}|Indlaw|\d{4}\s+\w+\s+\d+', content)),
            'contains_judgment': bool(re.search(r'held|judgment|order|conclusion', content, re.IGNORECASE)),
            'paragraph_numbers': self._extract_paragraph_numbers(content)
        })
        
        return TextChunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char
        )
    
    def _extract_paragraph_numbers(self, text: str) -> List[int]:
        """Extract paragraph numbers from text."""
        paragraph_numbers = []
        matches = re.findall(r'^(\d+)\.\s+', text, re.MULTILINE)
        for match in matches:
            try:
                paragraph_numbers.append(int(match))
            except ValueError:
                continue
        return sorted(paragraph_numbers)
    
    def chunk_document(self, processed_doc) -> List[TextChunk]:
        """
        Chunk a processed document.
        
        Args:
            processed_doc: ProcessedDocument object
        
        Returns:
            List of TextChunk objects
        """
        # Use cleaned content for chunking
        text = processed_doc.cleaned_content
        
        # Prepare metadata
        metadata = processed_doc.metadata.to_dict()
        metadata.update({
            'total_sections': len(processed_doc.sections),
            'original_length': len(processed_doc.content),
            'cleaned_length': len(text)
        })
        
        # Create chunks
        chunks = self.create_semantic_chunks(text, metadata)
        
        self.logger.debug(f"Created {len(chunks)} chunks for document {metadata['case_id']}")
        
        return chunks
    
    def chunk_documents_batch(self, processed_docs: List) -> List[TextChunk]:
        """
        Chunk multiple processed documents.
        
        Args:
            processed_docs: List of ProcessedDocument objects
        
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        total_docs = len(processed_docs)
        
        self.logger.info(f"Chunking {total_docs} documents...")
        
        for i, doc in enumerate(processed_docs):
            if i % 100 == 0:
                self.logger.info(f"Chunking progress: {i}/{total_docs} documents")
            
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} total chunks from {total_docs} documents")
        
        return all_chunks