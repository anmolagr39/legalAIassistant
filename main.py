import argparse
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from rag_engine import FIRRagEngine

def main():
    parser = argparse.ArgumentParser(description="FIR RAG System using ChromaDB and Gemini")
    parser.add_argument("--mode", choices=["ingest", "query"], required=True,
                       help="Mode: 'ingest' to load data, 'query' to ask questions")
    parser.add_argument("--csv", type=str, help="Path to FIR dataset CSV file")
    parser.add_argument("--query", type=str, help="Query to ask the system")
    parser.add_argument("--api-key", type=str, help="Google Gemini API key")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        Config.set_gemini_api_key(args.api_key)
    else:
        # Try to get from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            Config.set_gemini_api_key(api_key)
    
    # Initialize RAG engine
    try:
        rag_engine = FIRRagEngine(Config)
    except Exception as e:
        print(f"❌ Failed to initialize RAG engine: {e}")
        return
    
    if args.mode == "ingest":
        if not args.csv:
            print("❌ Please provide --csv path for ingest mode")
            return
        
        try:
            chunks_count = rag_engine.ingest_fir_data(args.csv)
            print(f"🎉 Successfully ingested {chunks_count} chunks!")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
    
    elif args.mode == "query":
        if not args.query:
            print("❌ Please provide --query for query mode")
            return
        
        if not Config.GEMINI_API_KEY:
            print("❌ Please provide Gemini API key with --api-key")
            return
        
        try:
            # Set top-k if provided
            if args.top_k:
                Config.TOP_K_RESULTS = args.top_k
            
            result = rag_engine.query(args.query)
            
            print("\n" + "="*60)
            print("📋 QUERY RESULTS")
            print("="*60)
            print(f"Query: {args.query}")
            print(f"Retrieved: {result['retrieved_count']} relevant documents")
            print("\n📝 ANSWER:")
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📚 SOURCES:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. Document: {source.get('doc_id', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")

if __name__ == "__main__":
    main()