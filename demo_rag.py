"""
Interactive RAG Demo Script
Run this script to test multiple queries against the FIR dataset
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_data_status():
    """Check if data is ingested"""
    try:
        from config import Config
        import chromadb
        
        client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        collection = client.get_collection(Config.COLLECTION_NAME)
        count = collection.count()
        print(f"✅ Found {count} documents in database")
        return count > 0
    except Exception as e:
        print(f"❌ No data found: {e}")
        return False

def ingest_data():
    """Ingest FIR dataset"""
    try:
        from config import Config
        from rag_engine import FIRRagEngine
        
        csv_path = "FIR_DATASET.csv"
        if not os.path.exists(csv_path):
            print(f"❌ FIR_DATASET.csv not found in current directory")
            return False
        
        print("🔄 Ingesting FIR dataset...")
        rag = FIRRagEngine(Config)
        chunks = rag.ingest_fir_data(csv_path)
        print(f"✅ Successfully ingested {chunks} chunks!")
        return True
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False

def run_query_demo():
    """Run interactive query demo"""
    try:
        from config import Config
        from rag_engine import FIRRagEngine
        
        # Check if Gemini API key is set
        if not Config.GEMINI_API_KEY:
            print("\n🔑 Gemini API Key Required")
            print("To use Gemini for generating responses, please provide your API key.")
            print("You can get one at: https://makersuite.google.com/app/apikey")
            
            while True:
                api_key = input("\nEnter your Gemini API key (or 'skip' for local-only mode): ").strip()
                if api_key.lower() == 'skip':
                    print("⚠️ Continuing without Gemini - using rule-based responses only")
                    break
                elif api_key:
                    Config.set_gemini_api_key(api_key)
                    print("✅ Gemini API key set!")
                    break
                else:
                    print("❌ Please enter a valid API key or type 'skip'")
        
        # Initialize RAG engine
        rag = FIRRagEngine(Config)
        
        # Sample queries for testing
        sample_queries = [
            "What is IPC Section 140?",
            "military uniform punishment",
            "section 127 stolen property",
            "cheating under IPC",
            "theft sections",
            "cognizable offenses",
            "non-bailable offenses",
            "assault punishment"
        ]
        
        print("\n" + "="*60)
        print("🚀 FIR RAG DEMO - Interactive Query Testing")
        print("="*60)
        print("\n📋 Sample queries you can try:")
        for i, query in enumerate(sample_queries, 1):
            print(f"  {i}. {query}")
        
        print("\n💡 Instructions:")
        print("- Type any query about IPC sections, laws, or legal matters")
        print("- Type 'sample' followed by a number (1-8) to use sample queries")
        print("- Type 'quit' or 'exit' to stop")
        
        while True:
            print("\n" + "-"*50)
            user_input = input("🔍 Enter your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Handle sample queries
            if user_input.lower().startswith('sample'):
                try:
                    parts = user_input.split()
                    if len(parts) == 2:
                        sample_num = int(parts[1])
                        if 1 <= sample_num <= len(sample_queries):
                            user_input = sample_queries[sample_num - 1]
                            print(f"🎯 Using sample query: {user_input}")
                        else:
                            print(f"❌ Sample number must be 1-{len(sample_queries)}")
                            continue
                    else:
                        print("❌ Use format: sample 1 (or sample 2, etc.)")
                        continue
                except ValueError:
                    print("❌ Invalid sample number")
                    continue
            
            if not user_input:
                print("❌ Please enter a query")
                continue
            
            try:
                print(f"🔄 Processing query: '{user_input}'")
                
                # Show which mode we're using
                if Config.GEMINI_API_KEY and hasattr(rag, 'gemini_model') and rag.gemini_model:
                    print("🤖 Using Gemini AI for response generation")
                else:
                    print("📋 Using rule-based analysis (no Gemini)")
                
                result = rag.query(user_input)
                
                print(f"\n📊 Results:")
                print(f"📋 Retrieved: {result['retrieved_count']} relevant documents")
                print(f"\n💬 Answer:")
                print("-" * 40)
                print(result['answer'])
                print("-" * 40)
                
                if result['sources']:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        doc_id = source.get('doc_id', 'Unknown')
                        row_idx = source.get('row_index', 'Unknown')
                        print(f"  {i}. Document: {doc_id} (Row: {row_idx})")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
                print("💡 Try a simpler query or check your setup")
        
    except Exception as e:
        print(f"❌ Demo setup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure data is ingested (run option 1 first)")
        print("2. Check if all dependencies are installed")
        print("3. Verify the RAG engine is properly configured")

def main():
    print("🚀 FIR RAG System Demo")
    print("=" * 30)
    
    # Check if data exists
    if not check_data_status():
        print("\n🔄 No data found. Let's ingest the FIR dataset first...")
        if not ingest_data():
            print("❌ Failed to ingest data. Please check the setup.")
            return
    
    # Run the demo
    run_query_demo()

if __name__ == "__main__":
    main()