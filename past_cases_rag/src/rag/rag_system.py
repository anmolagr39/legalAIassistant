"""
RAG query system with Gemini 2.5 Flash integration.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils import get_logger, GOOGLE_API_KEY, GEMINI_MODEL, GENERATION_CONFIG, MAX_RETRIEVAL_DOCS
from .vector_db import ChromaVectorDB

logger = get_logger(__name__)

class LegalRAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system for legal document Q&A.
    """
    
    def __init__(self, vector_db: Optional[ChromaVectorDB] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.vector_db = vector_db or ChromaVectorDB()
        
        # Initialize Gemini
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
        self.logger.info(f"RAG system initialized with model: {GEMINI_MODEL}")
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance user query with legal context and terminology.
        
        Args:
            query: Original user query
        
        Returns:
            Enhanced query string
        """
        # Add legal context terms for better retrieval
        legal_terms = [
            "Supreme Court", "judgment", "case", "ruling", "precedent",
            "legal", "court", "petition", "appeal", "order"
        ]
        
        # Simple query enhancement - add relevant legal terms
        enhanced_query = query
        
        # If query doesn't contain legal terms, add some context
        if not any(term.lower() in query.lower() for term in legal_terms):
            enhanced_query = f"legal case judgment {query}"
        
        return enhanced_query
    
    def retrieve_relevant_documents(self, query: str, num_docs: int = MAX_RETRIEVAL_DOCS,
                                  filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector database.
        
        Args:
            query: Search query
            num_docs: Number of documents to retrieve
            filters: Optional metadata filters
        
        Returns:
            List of relevant documents with metadata
        """
        try:
            enhanced_query = self.enhance_query(query)
            
            # Search in vector database
            results = self.vector_db.search(
                query=enhanced_query,
                n_results=num_docs,
                where_filter=filters
            )
            
            self.logger.debug(f"Retrieved {len(results)} documents for query: '{query}'")
            return results
            
        except Exception as e:
            self.logger.error(f"Document retrieval failed: {str(e)}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant legal documents found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get('metadata', {})
            content = doc.get('document', '')
            similarity = doc.get('similarity', 0)
            
            # Extract key metadata
            case_name = metadata.get('case_name', 'Unknown Case')
            court = metadata.get('court', 'Unknown Court')
            date = metadata.get('date', 'Unknown Date')
            judge = metadata.get('judge', 'Unknown Judge')
            
            # Format document section
            context_part = f"""
Document {i} (Relevance: {similarity:.2%}):
Case: {case_name}
Court: {court}
Date: {date}
Judge: {judge}

Content: {content[:800]}{'...' if len(content) > 800 else ''}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def create_legal_prompt(self, query: str, context: str) -> str:
        """
        Create a specialized prompt for legal Q&A.
        
        Args:
            query: User question
            context: Retrieved document context
        
        Returns:
            Formatted prompt for Gemini
        """
        prompt = f"""You are an expert legal assistant specializing in Indian Supreme Court cases. You help users understand legal precedents, case law, and judicial reasoning.

**Context from Supreme Court Cases:**
{context}

**User Question:** {query}

**Instructions:**
1. Answer the question based ONLY on the provided context from Supreme Court cases
2. If the context doesn't contain enough information, clearly state what information is missing
3. Always cite the specific cases you reference (case names, dates, courts)
4. Provide clear, accurate legal reasoning
5. Use appropriate legal terminology
6. If multiple cases are relevant, compare and contrast their holdings
7. Structure your answer clearly with headings if appropriate

**Answer:**"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Gemini 2.5 Flash.
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            Generated response string
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(**GENERATION_CONFIG)
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def answer_question(self, query: str, num_docs: int = MAX_RETRIEVAL_DOCS,
                       filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline to answer a legal question.
        
        Args:
            query: User question
            num_docs: Number of documents to retrieve
            filters: Optional metadata filters for search
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant documents
            self.logger.info(f"Processing query: '{query}'")
            retrieved_docs = self.retrieve_relevant_documents(query, num_docs, filters)
            
            if not retrieved_docs:
                return {
                    'query': query,
                    'answer': "I couldn't find any relevant Supreme Court cases for your question. Please try rephrasing or asking about a different legal topic.",
                    'sources': [],
                    'metadata': {
                        'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
                        'num_sources': 0,
                        'status': 'no_results'
                    }
                }
            
            # Step 2: Format context
            context = self.format_context(retrieved_docs)
            
            # Step 3: Create prompt
            prompt = self.create_legal_prompt(query, context)
            
            # Step 4: Generate response
            answer = self.generate_response(prompt)
            
            # Step 5: Format sources
            sources = []
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                sources.append({
                    'case_name': metadata.get('case_name', 'Unknown'),
                    'court': metadata.get('court', 'Unknown'),
                    'date': metadata.get('date', 'Unknown'),
                    'judge': metadata.get('judge', 'Unknown'),
                    'relevance': doc.get('similarity', 0),
                    'chunk_id': metadata.get('chunk_id', 'Unknown')
                })
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return {
                'query': query,
                'answer': answer,
                'sources': sources,
                'metadata': {
                    'processing_time_ms': processing_time,
                    'num_sources': len(sources),
                    'model_used': GEMINI_MODEL,
                    'status': 'success'
                }
            }
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {str(e)}")
            
            return {
                'query': query,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'metadata': {
                    'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
                    'num_sources': 0,
                    'status': 'error',
                    'error': str(e)
                }
            }
    
    def search_cases_by_topic(self, topic: str, num_cases: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cases related to a specific legal topic.
        
        Args:
            topic: Legal topic to search for
            num_cases: Number of cases to return
        
        Returns:
            List of relevant cases
        """
        query = f"cases related to {topic} legal precedent"
        results = self.retrieve_relevant_documents(query, num_cases)
        
        # Group by case_id to get unique cases
        unique_cases = {}
        for result in results:
            case_id = result['metadata'].get('case_id')
            if case_id and case_id not in unique_cases:
                unique_cases[case_id] = {
                    'case_name': result['metadata'].get('case_name'),
                    'court': result['metadata'].get('court'),
                    'date': result['metadata'].get('date'),
                    'judge': result['metadata'].get('judge'),
                    'relevance': result.get('similarity', 0),
                    'case_id': case_id
                }
        
        return list(unique_cases.values())
    
    def get_case_summary(self, case_name: str) -> Dict[str, Any]:
        """
        Get a summary of a specific case.
        
        Args:
            case_name: Name of the case to summarize
        
        Returns:
            Dictionary with case summary and details
        """
        # Search for the specific case
        results = self.vector_db.search_by_metadata({'case_name': case_name}, limit=10)
        
        if not results:
            return {
                'case_name': case_name,
                'summary': f"Case '{case_name}' not found in the database.",
                'status': 'not_found'
            }
        
        # Combine all chunks from this case
        case_content = []
        metadata = results[0]['metadata']  # Use metadata from first result
        
        for result in results:
            case_content.append(result['document'])
        
        full_content = " ".join(case_content)
        
        # Generate summary using Gemini
        summary_prompt = f"""Please provide a concise summary of this Supreme Court case:

Case: {case_name}
Court: {metadata.get('court', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}
Judge: {metadata.get('judge', 'Unknown')}

Case Content: {full_content[:2000]}{'...' if len(full_content) > 2000 else ''}

Please summarize:
1. Key facts
2. Legal issues
3. Court's holding/decision
4. Significance/precedent set

Summary:"""
        
        summary = self.generate_response(summary_prompt)
        
        return {
            'case_name': case_name,
            'court': metadata.get('court'),
            'date': metadata.get('date'), 
            'judge': metadata.get('judge'),
            'summary': summary,
            'status': 'success'
        }