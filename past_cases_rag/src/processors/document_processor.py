"""
Legal document processor for parsing Supreme Court case documents.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..utils import get_logger

logger = get_logger(__name__)

@dataclass
class CaseMetadata:
    """Structure for case metadata."""
    case_id: str
    case_name: str
    court: str
    date: str
    petition_number: str
    judge: str
    file_path: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'case_id': self.case_id,
            'case_name': self.case_name,
            'court': self.court,
            'date': self.date,
            'petition_number': self.petition_number,
            'judge': self.judge,
            'file_path': self.file_path
        }

@dataclass
class ProcessedDocument:
    """Structure for processed legal document."""
    metadata: CaseMetadata
    content: str
    cleaned_content: str
    sections: Dict[str, str]
    
class LegalDocumentProcessor:
    """
    Processor for Indian Supreme Court case documents.
    Handles parsing, cleaning, and metadata extraction.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # Regex patterns for extracting metadata
        self.patterns = {
            'case_name': r'^(.+?)\s+v\s+(.+?)(?:\n|$)',
            'court': r'(Supreme Court of India|High Court|District Court)',
            'date': r'(\d{1,2}\s+\w+\s+\d{4})',
            'petition': r'(Writ Petition|Civil Appeal|Criminal Appeal|Special Leave Petition).*?No\.?\s*(\d+.*?)(?:\n|of)',
            'judge': r'(?:The Judgment was delivered by|J\.|Justice)[\s:]+([^,\n]+)',
        }
    
    def extract_metadata(self, content: str, file_path: str) -> CaseMetadata:
        """
        Extract metadata from document content.
        
        Args:
            content: Raw document content
            file_path: Path to the source file
        
        Returns:
            CaseMetadata object with extracted information
        """
        lines = content.strip().split('\n')
        
        # Extract case name (usually first line)
        case_name = "Unknown Case"
        if lines:
            case_name_match = re.search(self.patterns['case_name'], lines[0], re.IGNORECASE)
            if case_name_match:
                petitioner = case_name_match.group(1).strip()
                respondent = case_name_match.group(2).strip()
                case_name = f"{petitioner} v {respondent}"
            else:
                case_name = lines[0].strip()
        
        # Extract court (usually second line or in content)
        court = "Unknown Court"
        court_match = re.search(self.patterns['court'], content, re.IGNORECASE)
        if court_match:
            court = court_match.group(1)
        
        # Extract date
        date = "Unknown Date"
        date_match = re.search(self.patterns['date'], content)
        if date_match:
            date = date_match.group(1)
        
        # Extract petition number
        petition_number = "Unknown Petition"
        petition_match = re.search(self.patterns['petition'], content, re.IGNORECASE)
        if petition_match:
            petition_type = petition_match.group(1)
            petition_no = petition_match.group(2)
            petition_number = f"{petition_type} No. {petition_no}"
        
        # Extract judge name
        judge = "Unknown Judge"
        judge_match = re.search(self.patterns['judge'], content, re.IGNORECASE)
        if judge_match:
            judge = judge_match.group(1).strip().rstrip(',').rstrip('.')
        
        # Generate case ID from filename
        case_id = Path(file_path).stem
        
        return CaseMetadata(
            case_id=case_id,
            case_name=case_name,
            court=court,
            date=date,
            petition_number=petition_number,
            judge=judge,
            file_path=str(file_path)
        )
    
    def clean_content(self, content: str) -> str:
        """
        Clean and normalize document content.
        
        Args:
            content: Raw document content
        
        Returns:
            Cleaned content string
        """
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove special characters but keep legal citations
        content = re.sub(r'[^\w\s\.,;:()\[\]/-]', '', content)
        
        # Normalize legal citations
        content = re.sub(r'(\d{4})\s*Indlaw\s*(\w+)\s*(\d+)', r'\1 Indlaw \2 \3', content)
        content = re.sub(r'AIR\s*(\d{4})\s*(\w+)\s*(\d+)', r'AIR \1 \2 \3', content)
        
        # Remove page numbers and other artifacts
        content = re.sub(r'Page\s+\d+', '', content)
        content = re.sub(r'\f', '', content)  # Remove form feeds
        
        return content.strip()
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract different sections from legal document.
        
        Args:
            content: Document content
        
        Returns:
            Dictionary of section names and their content
        """
        sections = {}
        lines = content.split('\n')
        
        current_section = "preamble"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a numbered paragraph (common in legal docs)
            if re.match(r'^\d+\.\s+', line):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                paragraph_num = re.match(r'^(\d+)\.', line).group(1)
                current_section = f"paragraph_{paragraph_num}"
                current_content = [line]
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def process_document(self, file_path: str) -> Optional[ProcessedDocument]:
        """
        Process a single legal document.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            ProcessedDocument object or None if processing fails
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                self.logger.warning(f"Empty document: {file_path}")
                return None
            
            # Extract metadata
            metadata = self.extract_metadata(content, file_path)
            
            # Clean content
            cleaned_content = self.clean_content(content)
            
            # Extract sections
            sections = self.extract_sections(cleaned_content)
            
            self.logger.debug(f"Processed document: {metadata.case_name}")
            
            return ProcessedDocument(
                metadata=metadata,
                content=content,
                cleaned_content=cleaned_content,
                sections=sections
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            return None
    
    def process_documents_batch(self, doc_paths: List[str]) -> List[ProcessedDocument]:
        """
        Process multiple documents in batch.
        
        Args:
            doc_paths: List of document file paths
        
        Returns:
            List of ProcessedDocument objects
        """
        processed_docs = []
        total_docs = len(doc_paths)
        
        self.logger.info(f"Processing {total_docs} documents...")
        
        for i, doc_path in enumerate(doc_paths):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{total_docs} documents processed")
            
            processed_doc = self.process_document(doc_path)
            if processed_doc:
                processed_docs.append(processed_doc)
        
        self.logger.info(f"Successfully processed {len(processed_docs)}/{total_docs} documents")
        return processed_docs
    
    def save_metadata_summary(self, processed_docs: List[ProcessedDocument], output_path: str):
        """
        Save metadata summary to JSON file.
        
        Args:
            processed_docs: List of processed documents
            output_path: Path to save the summary
        """
        summary = {
            'total_documents': len(processed_docs),
            'processing_date': datetime.now().isoformat(),
            'documents': [doc.metadata.to_dict() for doc in processed_docs]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Metadata summary saved to: {output_path}")