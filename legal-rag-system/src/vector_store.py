import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
from rich.console import Console
from rich.progress import Progress
import numpy as np

console = Console()

class VectorStore:
    """Manages ChromaDB vector store for legal documents."""
    
    def __init__(self, persist_directory: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        console.print(f"[blue]Loading embedding model: {embedding_model}[/blue]")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        console.print(f"[green]Vector store initialized at: {persist_directory}[/green]")
    
    def create_collection(self, collection_name: str, reset: bool = False) -> chromadb.Collection:
        """Create or get a ChromaDB collection."""
        try:
            if reset:
                try:
                    self.client.delete_collection(name=collection_name)
                    console.print(f"[yellow]Deleted existing collection: {collection_name}[/yellow]")
                except:
                    pass
            
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            console.print(f"[green]Collection '{collection_name}' ready[/green]")
            return collection
            
        except Exception as e:
            console.print(f"[red]Error creating collection {collection_name}: {e}[/red]")
            raise
    
    def add_documents(self, collection_name: str, chunks: List[Dict[str, Any]], 
                     batch_size: int = 100, reset: bool = False):
        """Add documents to ChromaDB collection."""
        collection = self.create_collection(collection_name, reset=reset)
        
        console.print(f"[blue]Adding {len(chunks)} documents to {collection_name}[/blue]")
        
        # Process in batches
        with Progress() as progress:
            task = progress.add_task("[green]Embedding documents...", total=len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Extract data for batch
                ids = [chunk['id'] for chunk in batch]
                documents = [chunk['content'] for chunk in batch]
                metadatas = [chunk['metadata'] for chunk in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(documents).tolist()
                
                # Add to collection
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                progress.update(task, advance=len(batch))
        
        console.print(f"[green]Successfully added {len(chunks)} documents to {collection_name}[/green]")
    
    def search(self, collection_name: str, query: str, top_k: int = 5, 
               metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=metadata_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Error searching collection {collection_name}: {e}[/red]")
            return []
    
    def hybrid_search(self, constitution_collection: str, legal_acts_collection: str, 
                     query: str, top_k: int = 5, constitution_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Perform hybrid search across both collections."""
        console.print(f"[blue]Performing hybrid search for: '{query}'[/blue]")
        
        # Search both collections
        const_results = self.search(constitution_collection, query, top_k)
        acts_results = self.search(legal_acts_collection, query, top_k)
        
        # Combine and weight results
        all_results = []
        
        # Add constitution results with weight
        for result in const_results:
            result['weighted_score'] = result['similarity'] * constitution_weight
            result['source_collection'] = constitution_collection
            all_results.append(result)
        
        # Add legal acts results with weight
        for result in acts_results:
            result['weighted_score'] = result['similarity'] * (1 - constitution_weight)
            result['source_collection'] = legal_acts_collection
            all_results.append(result)
        
        # Sort by weighted score and return top_k
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return all_results[:top_k]
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            
            return {
                'name': collection_name,
                'document_count': count,
                'status': 'active'
            }
        except Exception as e:
            return {
                'name': collection_name,
                'document_count': 0,
                'status': f'error: {e}'
            }
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            console.print(f"[red]Error listing collections: {e}[/red]")
            return []