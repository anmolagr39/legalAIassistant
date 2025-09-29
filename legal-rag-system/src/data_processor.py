import fitz  # PyMuPDF
import pandas as pd
import re
from typing import List, Dict, Any
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress
import os

console = Console()

class DataProcessor:
    """Handles processing of PDF and CSV data for the legal RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        console.print(f"[blue]Extracting text from PDF: {pdf_path}[/blue]")
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            with Progress() as progress:
                task = progress.add_task("[green]Processing pages...", total=len(doc))
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                    progress.update(task, advance=1)
            
            doc.close()
            console.print(f"[green]Successfully extracted {len(text)} characters from PDF[/green]")
            return text
            
        except Exception as e:
            console.print(f"[red]Error extracting PDF text: {e}[/red]")
            return ""
    
    def process_constitution_text(self, text: str) -> List[Dict[str, Any]]:
        """Process constitution text and split by articles."""
        console.print("[blue]Processing Constitution text...[/blue]")
        
        chunks = []
        
        # Split by articles (Article followed by number)
        article_pattern = r'(Article\s+\d+[A-Z]*\.?\s*[—\-]?\s*[^\n]*)'
        articles = re.split(article_pattern, text, flags=re.IGNORECASE)
        
        current_article = None
        current_content = ""
        
        article_counter = 0
        for i, segment in enumerate(articles):
            if re.match(r'Article\s+\d+', segment, re.IGNORECASE):
                # Save previous article if exists
                if current_article and current_content.strip():
                    article_chunks = self._chunk_text(current_content.strip())
                    for j, chunk in enumerate(article_chunks):
                        chunks.append({
                            'id': f"const_art_{article_counter}_{j}",
                            'content': chunk,
                            'metadata': {
                                'source': 'constitution',
                                'article': current_article,
                                'chunk_index': j,
                                'type': 'article'
                            }
                        })
                    article_counter += 1
                
                current_article = segment.strip()
                current_content = ""
            else:
                current_content += segment
        
        # Process the last article
        if current_article and current_content.strip():
            article_chunks = self._chunk_text(current_content.strip())
            for j, chunk in enumerate(article_chunks):
                chunks.append({
                    'id': f"const_art_{article_counter}_{j}",
                    'content': chunk,
                    'metadata': {
                        'source': 'constitution',
                        'article': current_article,
                        'chunk_index': j,
                        'type': 'article'
                    }
                })
            article_counter += 1
        
        # If no articles found, chunk the entire text
        if not chunks:
            text_chunks = self._chunk_text(text)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'id': f"const_general_{i}",
                    'content': chunk,
                    'metadata': {
                        'source': 'constitution',
                        'chunk_index': i,
                        'type': 'general'
                    }
                })
        
        # Ensure all IDs are unique
        seen_ids = set()
        for i, chunk in enumerate(chunks):
            original_id = chunk['id']
            counter = 0
            while chunk['id'] in seen_ids:
                counter += 1
                chunk['id'] = f"{original_id}_dup_{counter}"
            seen_ids.add(chunk['id'])
        
        console.print(f"[green]Created {len(chunks)} chunks from Constitution[/green]")
        return chunks
    
    def process_legal_acts_csv(self, csv_path: str, max_records: int = None) -> List[Dict[str, Any]]:
        """Process legal acts CSV file."""
        console.print(f"[blue]Processing legal acts CSV: {csv_path}[/blue]")
        
        try:
            # Read CSV with error handling
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
            
            if max_records:
                df = df.head(max_records)
                console.print(f"[yellow]Limited to {max_records} records for processing[/yellow]")
            
            console.print(f"[green]Loaded {len(df)} legal acts[/green]")
            
            chunks = []
            
            with Progress() as progress:
                task = progress.add_task("[green]Processing acts...", total=len(df))
                
                for idx, row in df.iterrows():
                    try:
                        # Extract data
                        enactment_date = str(row.get('Enactment Date', ''))
                        act_number = str(row.get('Act Number', ''))
                        short_title = str(row.get('Short Title', ''))
                        entity = str(row.get('Entity', ''))
                        markdown_content = str(row.get('Markdown', ''))
                        
                        # Skip if no content
                        if not markdown_content or markdown_content == 'nan':
                            progress.update(task, advance=1)
                            continue
                        
                        # Clean markdown content
                        cleaned_content = self._clean_markdown_content(markdown_content)
                        
                        # Chunk the content
                        act_chunks = self._chunk_text(cleaned_content)
                        
                        for j, chunk in enumerate(act_chunks):
                            chunks.append({
                                'id': f"act_{idx}_{j}",
                                'content': chunk,
                                'metadata': {
                                    'source': 'legal_acts',
                                    'enactment_date': enactment_date,
                                    'act_number': act_number,
                                    'short_title': short_title,
                                    'entity': entity,
                                    'act_index': idx,
                                    'chunk_index': j,
                                    'type': 'legal_act'
                                }
                            })
                    
                    except Exception as e:
                        console.print(f"[red]Error processing row {idx}: {e}[/red]")
                    
                    progress.update(task, advance=1)
            
            console.print(f"[green]Created {len(chunks)} chunks from legal acts[/green]")
            return chunks
            
        except Exception as e:
            console.print(f"[red]Error processing legal acts CSV: {e}[/red]")
            return []
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings in the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _clean_markdown_content(self, content: str) -> str:
        """Clean markdown content."""
        # Remove markdown headers
        content = re.sub(r'#+\s*', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Remove markdown formatting
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Italic
        content = re.sub(r'`(.*?)`', r'\1', content)        # Code
        
        return content.strip()
    
    def save_processed_data(self, chunks: List[Dict[str, Any]], filename: str):
        """Save processed chunks to file."""
        output_path = os.path.join("data", "processed", filename)
        
        # Convert to DataFrame for easy saving
        df_data = []
        for chunk in chunks:
            row = {
                'id': chunk['id'],
                'content': chunk['content']
            }
            # Flatten metadata
            for key, value in chunk['metadata'].items():
                row[f'metadata_{key}'] = value
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        console.print(f"[green]Saved {len(chunks)} chunks to {output_path}[/green]")