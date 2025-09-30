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
        
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} records")
        
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
        all_text = " ".join(documents)
        
        # Extract IPC sections
        ipc_sections = self._extract_ipc_sections(all_text)
        
        # Common cheating-related sections
        cheating_sections = {
            '420': 'Cheating and dishonestly inducing delivery of property',
            '415': 'Cheating',
            '406': 'Criminal breach of trust',
            '467': 'Forgery of valuable security',
            '468': 'Forgery for purpose of cheating',
            '471': 'Using as genuine a forged document',
            '419': 'Cheating by personation',
            '463': 'Forgery'
        }
        
        # Find relevant sections
        relevant_sections = []
        for section in ipc_sections:
            if section in cheating_sections:
                relevant_sections.append(f"IPC {section}: {cheating_sections[section]}")
        
        # If no specific sections found, provide common cheating sections
        if not relevant_sections:
            relevant_sections = [
                "IPC 420: Cheating and dishonestly inducing delivery of property",
                "IPC 415: Cheating",
                "IPC 419: Cheating by personation"
            ]
        
        # Extract key information from documents
        key_info = []
        for doc in documents[:3]:  # Analyze first 3 documents
            if 'cheat' in doc.lower() or 'fraud' in doc.lower():
                # Extract relevant parts
                lines = doc.split('|')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['cheat', 'fraud', 'deceiv', 'dishonest']):
                        key_info.append(line.strip())
        
        # Build response
        response_parts = []
        response_parts.append("**IPC Sections for Cheating:**")
        for section in relevant_sections[:5]:  # Limit to 5 sections
            response_parts.append(f"• {section}")
        
        if key_info:
            response_parts.append("\n**Relevant Case Information:**")
            for info in key_info[:3]:  # Limit to 3 pieces of info
                response_parts.append(f"• {info}")
        
        return "\n".join(response_parts)
    
    def generate_legal_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using Gemini or fallback analysis"""
        
        # If Gemini is available, try to use it with minimal tokens
        if self.config.GEMINI_API_KEY and self.gemini_model:
            try:
                # Very short prompt to conserve quota
                context = context_docs[0][:500] if context_docs else "No context"
                prompt = f"IPC sections for cheating based on: {context[:200]}"
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 100  # Very limited to save quota
                    }
                )
                
                if response and response.text:
                    return response.text
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print("⚠️ Quota exceeded, using fallback analysis")
                else:
                    print(f"⚠️ Generation failed: {e}")
        
        # Fallback: Use rule-based analysis
        print("📋 Using rule-based analysis (Gemini unavailable)")
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