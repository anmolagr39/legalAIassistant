import os
import sys
import pandas as pd
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import time
from typing import List, Dict, Optional, Tuple

from config import Config

class FIRRagEngine:
    """RAG Engine for FIR Dataset using ChromaDB and Google Gemini"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize ChromaDB and Gemini clients"""
        # Setup ChromaDB with absolute path
        chroma_path = self.config.get_absolute_path(self.config.CHROMA_DB_PATH)
        if not os.path.exists(chroma_path):
            os.makedirs(chroma_path)
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Setup Gemini with better error handling
        if self.config.GEMINI_API_KEY:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.gemini_model = None
            
            try:
                print("🔍 Checking available models...")
                models = genai.list_models()
                available_models = []
                
                for model in models:
                    if 'generateContent' in model.supported_generation_methods:
                        available_models.append(model.name)
                
                if available_models:
                    # Try models starting with lighter/cheaper ones
                    preferred_models = [
                        'models/gemini-2.0-flash-lite',
                        'models/gemini-2.0-flash',
                        'models/gemini-2.5-flash-lite',
                        'models/gemini-2.5-flash',
                        'models/gemini-flash-latest'
                    ]
                    
                    # Find the first preferred model that's available
                    model_to_use = None
                    for preferred in preferred_models:
                        if preferred in available_models:
                            model_to_use = preferred
                            break
                    
                    # If no preferred model found, use the first available
                    if not model_to_use:
                        model_to_use = available_models[0]
                    
                    print(f"🧪 Testing model: {model_to_use}")
                    
                    # Test with minimal content to conserve quota
                    test_model = genai.GenerativeModel(model_to_use)
                    test_response = test_model.generate_content(
                        "Hi",
                        generation_config={'max_output_tokens': 10}
                    )
                    
                    self.gemini_model = test_model
                    print(f"✅ Successfully initialized Gemini model: {model_to_use}")
                    
                else:
                    print("❌ No models support content generation")
                    
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    print("⚠️ Gemini API quota exceeded. Switching to fallback mode.")
                    print("   - Your free tier quota has been reached")
                    print("   - Responses will be based on retrieved documents only")
                    print("   - Wait 24 hours for quota reset or upgrade to paid plan")
                else:
                    print(f"❌ Error setting up Gemini: {e}")
                
                self.gemini_model = None
        else:
            print("⚠️ Gemini API key not set. Generation will be disabled.")
            self.gemini_model = None
        
        # Setup embedding model
        try:
            self.embedder = SentenceTransformer(self.config.EMBEDDING_MODEL)
            print(f"✅ RAG Engine initialized with embedding model: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def preprocess_fir_text(self, row: pd.Series) -> str:
        """Preprocess FIR data row into searchable text"""
        text_parts = []
        for column, value in row.items():
            if pd.notna(value) and str(value).strip():
                text_parts.append(f"{column}: {str(value)}")
        return " | ".join(text_parts)
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Split text into chunks for better retrieval"""
        if not text or len(text) < 50:
            return [{"chunk_id": f"{doc_id}_0", "text": text, "doc_id": doc_id}]
        
        chunks = []
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP
        
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_num}",
                "text": chunk_text,
                "doc_id": doc_id
            })
            
            start += chunk_size - overlap
            chunk_num += 1
        
        return chunks
    
    def ingest_fir_data(self, csv_path: str) -> int:
        """Ingest FIR dataset into ChromaDB"""
        print(f"📂 Loading FIR dataset from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        # Load dataset with proper encoding handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-16', quotechar='"', escapechar='\\')
        except UnicodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin-1', quotechar='"', escapechar='\\')
            except:
                df = pd.read_csv(csv_path, encoding='utf-8', quotechar='"', escapechar='\\')
        except pd.errors.ParserError:
            # Try with different quote handling
            try:
                df = pd.read_csv(csv_path, encoding='utf-16', quoting=1)  # QUOTE_ALL
            except:
                df = pd.read_csv(csv_path, encoding='utf-16', sep=',', on_bad_lines='skip')
        
        print(f"📊 Loaded {len(df)} records with {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Get or create collection
        try:
            collection = self.chroma_client.get_collection(self.config.COLLECTION_NAME)
        except:
            collection = self.chroma_client.create_collection(self.config.COLLECTION_NAME)
        
        # Process data in batches
        batch_size = 50
        total_chunks = 0
        
        for batch_start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            batch_texts = []
            batch_ids = []
            batch_metadata = []
            
            for idx, row in batch_df.iterrows():
                text = self.preprocess_fir_text(row)
                doc_id = f"fir_{idx}"
                chunks = self.chunk_text(text, doc_id)
                
                for chunk in chunks:
                    batch_texts.append(chunk['text'])
                    batch_ids.append(chunk['chunk_id'])
                    batch_metadata.append({
                        'doc_id': chunk['doc_id'],
                        'row_index': int(idx),
                        'source': 'fir_dataset'
                    })
            
            if batch_texts:
                embeddings = self.embedder.encode(batch_texts, convert_to_tensor=True)
                collection.add(
                    embeddings=embeddings.cpu().numpy().tolist(),
                    documents=batch_texts,
                    ids=batch_ids,
                    metadatas=batch_metadata
                )
                total_chunks += len(batch_texts)
        
        print(f"✅ Successfully ingested {total_chunks} text chunks")
        return total_chunks
    
    def retrieve_relevant_cases(self, query: str, top_k: Optional[int] = None) -> Dict:
        """Retrieve relevant FIR cases based on query"""
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        try:
            collection = self.chroma_client.get_collection(self.config.COLLECTION_NAME)
        except:
            raise Exception(f"Collection '{self.config.COLLECTION_NAME}' not found. Please ingest data first.")
        
        query_embedding = self.embedder.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return results
    
    def _extract_ipc_sections(self, text: str) -> List[str]:
        """Extract IPC sections from text"""
        # Look for IPC section patterns
        ipc_patterns = [
            r'(?:IPC|Indian Penal Code)\s*(?:Section|Sec\.?|§)?\s*(\d+[A-Z]?)',
            r'Section\s*(\d+[A-Z]?)',
            r'Sec\.?\s*(\d+[A-Z]?)',
            r'§\s*(\d+[A-Z]?)',
            r'(\d{2,3}[A-Z]?)\s*(?:IPC|Indian Penal Code)'
        ]
        
        sections = []
        for pattern in ipc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend(matches)
        
        return list(set(sections))  # Remove duplicates
    
    def _analyze_cheating_cases(self, documents: List[str]) -> str:
        """Analyze documents for cheating-related information"""
        # Generic document analysis: extract IPC sections, top matches to query-like terms,
        # and return a concise, neutral summary. This replaces the old cheating-specific
        # analyzer so the system can handle arbitrary legal queries.
        all_text = " ".join(documents)

        # Extract IPC sections present in the retrieved documents
        ipc_sections = self._extract_ipc_sections(all_text)

        # Gather short snippets that include likely legal keywords to surface context
        keywords = ['cheat', 'fraud', 'theft', 'assault', 'punish', 'sentence', 'bailable', 'cognizable', 'offence', 'section']
        snippets = []
        for doc in documents[:6]:
            # split by pipe or sentence boundaries to find compact snippets
            parts = re.split(r'\||\.|;|\n', doc)
            for part in parts:
                lower = part.lower()
                if any(k in lower for k in keywords):
                    snippet = part.strip()
                    if snippet and snippet not in snippets:
                        snippets.append(snippet)
                if len(snippets) >= 6:
                    break
            if len(snippets) >= 6:
                break

        # Build a neutral structured response
        response_parts = []
        if ipc_sections:
            response_parts.append("IPC sections found:")
            for s in sorted(ipc_sections)[:10]:
                response_parts.append(f"• IPC {s}")
        else:
            response_parts.append("No explicit IPC section numbers were found in the retrieved documents.")

        if snippets:
            response_parts.append("\nRepresentative snippets from retrieved documents:")
            for snip in snippets[:5]:
                response_parts.append(f"• {snip}")

        # If empty, provide a gentle guidance message
        if not ipc_sections and not snippets:
            response_parts.append("\nNo specific legal phrases detected. Provide a more targeted query (e.g., 'IPC 420 cheating').")

        return "\n".join(response_parts)
    
    def generate_legal_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using Gemini or fallback analysis"""
        # Build a succinct prompt using the user's query and a short context snippet.
        # This avoids hardcoding any specific offense domain.
        short_context = "".join(context_docs[:2])[:800] if context_docs else "No context available"
        prompt = (
            f"You are a legal assistant. Answer the user's query concisely using the retrieved context. "
            f"User query: {query}\nContext: {short_context}\nProvide:\n- Key IPC sections found (if any)\n- Short summary relevant to the query\n- Up to 3 representative snippets from the context."
        )

        # If Gemini is available, try to use it with low token usage
        if self.config.GEMINI_API_KEY and self.gemini_model:
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 200
                    }
                )
                if response and getattr(response, 'text', None):
                    return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print("⚠️ Gemini quota or rate limit; falling back to local analysis")
                else:
                    print(f"⚠️ Gemini generation error: {e}")

        # Local fallback: use the generic analyzer which is query-agnostic
        print("📋 Using local rule-based analysis (Gemini unavailable or fallback)")
        return self._analyze_cheating_cases(context_docs)
    
    def query(self, user_query: str) -> Dict:
        """Complete RAG pipeline: retrieve + generate"""
        print(f"🔍 Searching for: {user_query}")
        
        results = self.retrieve_relevant_cases(user_query)
        
        if not results['documents'] or not results['documents'][0]:
            return {
                'answer': "No relevant cases found in the database.",
                'sources': [],
                'retrieved_count': 0
            }
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        answer = self.generate_legal_response(user_query, documents)
        
        return {
            'answer': answer,
            'sources': metadatas,
            'retrieved_count': len(documents)
        }