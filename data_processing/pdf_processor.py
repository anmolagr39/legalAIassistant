"""
PDF processing for Constitution and Legal Acts
"""
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)


class ConstitutionProcessor:
    """Process Constitution of India PDF"""
    
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
    
    def extract_text(self) -> str:
        """Extract text from PDF"""
        logger.info(f"Extracting text from {self.pdf_path}")
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from Constitution PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            # Fallback to PyPDF2
            return self._extract_with_pypdf2()
    
    def _extract_with_pypdf2(self) -> str:
        """Fallback PDF extraction using PyPDF2"""
        logger.info("Using PyPDF2 for extraction")
        text = ""
        
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text
    
    def extract_articles(self, text: str, max_chars: int = 50000) -> List[Dict]:
        """Extract constitutional articles from text using LLM"""
        logger.info("Extracting constitutional articles using LLM-based approach")
        
        # Split text into chunks for processing
        chunk_size = 15000  # ~15K chars per chunk
        overlap = 1000
        text_chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            text_chunks.append(chunk)
        
        logger.info(f"Split text into {len(text_chunks)} chunks for article detection")
        
        # Store the full text for later detailed extraction
        self.full_text = text
        
        # Return empty list - articles will be extracted via LLM in main.py
        # We just need to prepare the text
        return []
    
    def _extract_articles_alternative(self, text: str) -> List[Dict]:
        """Alternative article extraction method"""
        articles = []
        
        # Split by common patterns
        lines = text.split('\n')
        current_article = None
        current_content = []
        
        for line in lines:
            # Check if line starts with article number
            match = re.match(r'^(\d+[A-Z]?)\.?\s+(.+)', line.strip())
            if match and len(match.group(1)) <= 4:  # Article numbers are typically short
                # Save previous article
                if current_article:
                    articles.append({
                        'article_number': current_article,
                        'title': current_content[0] if current_content else "",
                        'content': '\n'.join(current_content[1:])[:2000],
                        'full_text': '\n'.join(current_content)[:5000]
                    })
                
                # Start new article
                current_article = match.group(1)
                current_content = [match.group(2)]
            elif current_article:
                current_content.append(line.strip())
        
        # Add last article
        if current_article and current_content:
            articles.append({
                'article_number': current_article,
                'title': current_content[0] if current_content else "",
                'content': '\n'.join(current_content[1:])[:2000],
                'full_text': '\n'.join(current_content)[:5000]
            })
        
        logger.info(f"Alternative method extracted {len(articles)} articles")
        return articles
    
    def extract_parts(self, text: str) -> List[Dict]:
        """Extract constitutional parts"""
        parts = []
        
        # Pattern: PART I, PART II, etc.
        part_pattern = r'PART\s+([IVX]+)\.?\s*([^\n]+)'
        matches = re.finditer(part_pattern, text, re.IGNORECASE)
        
        for match in matches:
            part_num = match.group(1)
            part_title = match.group(2).strip()
            
            parts.append({
                'part_number': part_num,
                'title': part_title[:200]
            })
        
        logger.info(f"Extracted {len(parts)} constitutional parts")
        return parts
    
    def process(self) -> Dict:
        """Process entire constitution document"""
        text = self.extract_text()
        articles = self.extract_articles(text)
        parts = self.extract_parts(text)
        
        return {
            'full_text': text,
            'articles': articles,
            'parts': parts,
            'stats': {
                'total_characters': len(text),
                'total_articles': len(articles),
                'total_parts': len(parts)
            }
        }


def test_pdf_processing():
    """Test PDF processing"""
    from config.settings import CONSTITUTION_PDF
    
    if not CONSTITUTION_PDF.exists():
        print(f"Constitution PDF not found at {CONSTITUTION_PDF}")
        return
    
    print("\n=== Testing Constitution PDF Processing ===")
    processor = ConstitutionProcessor(CONSTITUTION_PDF)
    
    result = processor.process()
    
    print(f"Total characters: {result['stats']['total_characters']:,}")
    print(f"Total articles: {result['stats']['total_articles']}")
    print(f"Total parts: {result['stats']['total_parts']}")
    
    if result['articles']:
        print(f"\nSample Article:")
        sample = result['articles'][0]
        print(f"  Article {sample['article_number']}: {sample['title']}")
        print(f"  Content: {sample['content'][:200]}...")


if __name__ == "__main__":
    test_pdf_processing()
