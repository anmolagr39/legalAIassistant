import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from rich.console import Console

console = Console()

class LegalRAG:
    """Main RAG system for legal document queries."""
    
    def __init__(self, vector_store, constitution_collection: str, 
                 legal_acts_collection: str, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.constitution_collection = constitution_collection
        self.legal_acts_collection = legal_acts_collection
        
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        console.print("[green]Legal RAG system initialized[/green]")
    
    def query(self, question: str, search_type: str = "hybrid", top_k: int = 5, 
              include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the legal RAG system.
        
        Args:
            question: User's legal question
            search_type: "hybrid", "constitution", "legal_acts"
            top_k: Number of relevant documents to retrieve
            include_sources: Whether to include source information
        """
        console.print(f"[blue]Processing query: '{question}'[/blue]")
        
        # Retrieve relevant documents
        if search_type == "hybrid":
            retrieved_docs = self.vector_store.hybrid_search(
                self.constitution_collection, 
                self.legal_acts_collection, 
                question, 
                top_k
            )
        elif search_type == "constitution":
            retrieved_docs = self.vector_store.search(
                self.constitution_collection, 
                question, 
                top_k
            )
        elif search_type == "legal_acts":
            retrieved_docs = self.vector_store.search(
                self.legal_acts_collection, 
                question, 
                top_k
            )
        else:
            raise ValueError("search_type must be 'hybrid', 'constitution', or 'legal_acts'")
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'query': question
            }
        
        # Generate response using Gemini
        response = self._generate_response(question, retrieved_docs)
        
        # Prepare sources
        sources = []
        if include_sources:
            sources = self._format_sources(retrieved_docs)
        
        return {
            'answer': response,
            'sources': sources,
            'query': question,
            'retrieved_docs_count': len(retrieved_docs)
        }
    
    def _generate_response(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini API."""
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_type = doc['metadata'].get('source', 'unknown')
            
            if source_type == 'constitution':
                article = doc['metadata'].get('article', 'Unknown Article')
                context_parts.append(f"[Constitution - {article}]:\n{doc['content']}\n")
            elif source_type == 'legal_acts':
                act_title = doc['metadata'].get('short_title', 'Unknown Act')
                act_date = doc['metadata'].get('enactment_date', 'Unknown Date')
                context_parts.append(f"[Legal Act - {act_title} ({act_date})]:\n{doc['content']}\n")
            else:
                context_parts.append(f"[Document {i}]:\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an expert legal assistant with deep knowledge of Indian law, including the Constitution of India and various legal acts. 

Based on the following legal documents, please provide a comprehensive and accurate answer to the user's question. 

IMPORTANT GUIDELINES:
1. Base your answer strictly on the provided legal documents
2. Cite specific articles, sections, or acts when relevant
3. If the documents don't contain sufficient information, clearly state this
4. Provide clear, structured responses
5. Use legal terminology appropriately
6. If there are multiple relevant provisions, explain them clearly
7. Distinguish between constitutional provisions and statutory provisions

LEGAL DOCUMENTS:
{context}

USER QUESTION: {question}

Please provide a detailed legal analysis and answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            console.print(f"[red]Error generating response: {e}[/red]")
            return f"I encountered an error while generating the response: {e}"
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for display."""
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc['metadata']
            source_type = metadata.get('source', 'unknown')
            
            source_info = {
                'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                'similarity_score': round(doc.get('similarity', 0), 3),
                'source_type': source_type
            }
            
            if source_type == 'constitution':
                source_info.update({
                    'article': metadata.get('article', 'Unknown'),
                    'type': metadata.get('type', 'article')
                })
            elif source_type == 'legal_acts':
                source_info.update({
                    'act_title': metadata.get('short_title', 'Unknown Act'),
                    'enactment_date': metadata.get('enactment_date', 'Unknown'),
                    'act_number': metadata.get('act_number', 'Unknown'),
                    'entity': metadata.get('entity', 'Unknown')
                })
            
            sources.append(source_info)
        
        return sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        const_stats = self.vector_store.get_collection_stats(self.constitution_collection)
        acts_stats = self.vector_store.get_collection_stats(self.legal_acts_collection)
        
        return {
            'constitution_collection': const_stats,
            'legal_acts_collection': acts_stats,
            'total_documents': const_stats['document_count'] + acts_stats['document_count'],
            'available_collections': self.vector_store.list_collections()
        }
    
    def search_by_metadata(self, collection: str, metadata_filter: Dict[str, str], 
                          query: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents by metadata filters."""
        
        if query:
            # Semantic search with metadata filter
            return self.vector_store.search(collection, query, top_k, metadata_filter)
        else:
            # Get collection and filter by metadata only
            try:
                collection_obj = self.vector_store.client.get_collection(name=collection)
                results = collection_obj.get(
                    where=metadata_filter,
                    limit=top_k,
                    include=['documents', 'metadatas']
                )
                
                formatted_results = []
                for i in range(len(results['ids'])):
                    formatted_results.append({
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    })
                
                return formatted_results
                
            except Exception as e:
                console.print(f"[red]Error in metadata search: {e}[/red]")
                return []