"""
Data preprocessing for legal documents
"""
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class LegalTextPreprocessor:
    """Preprocess legal text for entity extraction"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal punctuation
        # Keep: periods, commas, colons, semicolons, parentheses, hyphens
        text = re.sub(r'[^\w\s.,;:()\-/&\'\"]+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_section_number(text: str) -> str:
        """Extract IPC section number from text"""
        # Pattern: Section 123, Section 123A, s. 123, sec 123
        patterns = [
            r'(?:Section|section|Sec|sec|s\.)\s*(\d+[A-Z]?)',
            r'IPC\s+(\d+[A-Z]?)',
            r'§\s*(\d+[A-Z]?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def extract_case_number(text: str) -> str:
        """Extract case number from text"""
        # Pattern: Writ Petition No. 117 of 1973, Criminal Appeal No. 123/2020
        patterns = [
            r'(?:Writ Petition|Criminal Appeal|Civil Appeal|SLP|Special Leave Petition)\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'(?:Writ Petition|Criminal Appeal|Civil Appeal)\s+No\.?\s*(\d+/\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    @staticmethod
    def extract_court_name(text: str) -> str:
        """Extract court name from text"""
        courts = [
            "Supreme Court of India",
            "Supreme Court",
            "High Court",
            "District Court",
            "Sessions Court",
            "Magistrate Court"
        ]
        
        for court in courts:
            if court.lower() in text.lower():
                return court
        
        return None
    
    @staticmethod
    def extract_date(text: str) -> str:
        """Extract date from text"""
        # Pattern: 26 September 1973, 25-12-1949, 26/01/1950
        patterns = [
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def extract_article_numbers(text: str) -> List[str]:
        """Extract constitutional article numbers"""
        # Pattern: Article 14, Art. 21, Art 19(1)(f)
        pattern = r'(?:Article|Art\.?|art\.?)\s*(\d+(?:\([^)]+\))?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))  # Remove duplicates
    
    @staticmethod
    def chunk_text(text: str, max_length: int = 4000, overlap: int = 200) -> List[str]:
        """Chunk text into smaller pieces with overlap"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period followed by space and capital letter
                boundary = text.rfind('. ', start, end)
                if boundary > start:
                    end = boundary + 1
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else end
        
        return chunks


class FIRDatasetLoader:
    """Load and preprocess FIR dataset"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.preprocessor = LegalTextPreprocessor()
    
    def load(self) -> pd.DataFrame:
        """Load FIR dataset"""
        logger.info(f"Loading FIR dataset from {self.file_path}")
        df = pd.read_csv(self.file_path)
        logger.info(f"Loaded {len(df)} IPC sections")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> List[Dict]:
        """Preprocess FIR dataset into structured format"""
        processed = []
        
        for idx, row in df.iterrows():
            section_data = {
                'section_number': self.preprocessor.extract_section_number(row['URL']) or f"Section_{idx}",
                'description': self.preprocessor.clean_text(row.get('Description', '')),
                'offense': self.preprocessor.clean_text(row.get('Offense', '')),
                'punishment': self.preprocessor.clean_text(row.get('Punishment', '')),
                'cognizable': str(row.get('Cognizable', '')),
                'bailable': str(row.get('Bailable', '')),
                'court': str(row.get('Court', '')),
                'url': str(row.get('URL', ''))
            }
            processed.append(section_data)
        
        logger.info(f"Preprocessed {len(processed)} IPC sections")
        return processed


class CaseDocumentLoader:
    """Load and preprocess case documents"""
    
    def __init__(self, docs_dir: Path, max_cases: int = 300):
        self.docs_dir = docs_dir
        self.max_cases = max_cases
        self.preprocessor = LegalTextPreprocessor()
    
    def load(self) -> List[Dict]:
        """Load case documents"""
        logger.info(f"Loading case documents from {self.docs_dir}")
        
        case_files = sorted(list(self.docs_dir.glob("*.txt")))[:self.max_cases]
        logger.info(f"Found {len(case_files)} case files (limited to {self.max_cases})")
        
        cases = []
        for file_path in case_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                case_data = {
                    'case_id': file_path.stem,
                    'file_path': str(file_path),
                    'content': content,
                    'content_length': len(content)
                }
                cases.append(case_data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(cases)} case documents")
        return cases
    
    def preprocess(self, cases: List[Dict]) -> List[Dict]:
        """Preprocess case documents"""
        processed = []
        
        for case in cases:
            content = case['content']
            
            # Extract metadata
            case_name = content.split('\n')[0].strip() if content else "Unknown"
            court = self.preprocessor.extract_court_name(content[:500])
            case_number = self.preprocessor.extract_case_number(content[:1000])
            date = self.preprocessor.extract_date(content[:1000])
            
            # Clean content
            cleaned_content = self.preprocessor.clean_text(content)
            
            # Chunk if too long
            chunks = self.preprocessor.chunk_text(cleaned_content, max_length=6000)
            
            processed_case = {
                'case_id': case['case_id'],
                'case_name': case_name[:200],
                'case_number': case_number,
                'court': court,
                'date': date,
                'content': cleaned_content,
                'chunks': chunks,
                'num_chunks': len(chunks)
            }
            processed.append(processed_case)
        
        logger.info(f"Preprocessed {len(processed)} cases")
        return processed


def test_preprocessing():
    """Test preprocessing functions"""
    from config.settings import FIR_DATASET, CASE_DOCS_DIR, MAX_CASES
    
    # Test FIR dataset
    print("\n=== Testing FIR Dataset Preprocessing ===")
    fir_loader = FIRDatasetLoader(FIR_DATASET)
    df = fir_loader.load()
    processed_fir = fir_loader.preprocess(df.head(5))
    print(f"Sample IPC Section: {processed_fir[0]}")
    
    # Test case documents
    print("\n=== Testing Case Document Preprocessing ===")
    case_loader = CaseDocumentLoader(CASE_DOCS_DIR, max_cases=5)
    cases = case_loader.load()
    processed_cases = case_loader.preprocess(cases)
    print(f"Sample Case: {processed_cases[0]['case_name']}")
    print(f"  Court: {processed_cases[0]['court']}")
    print(f"  Chunks: {processed_cases[0]['num_chunks']}")


if __name__ == "__main__":
    test_preprocessing()
