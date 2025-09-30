import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def run_ingest_mode():
    """Run data ingestion"""
    from config import Config
    from rag_engine import FIRRagEngine
    
    csv_path = input("Enter path to your FIR dataset CSV: ").strip()
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    try:
        rag = FIRRagEngine(Config)
        chunks = rag.ingest_fir_data(csv_path)
        print(f"🎉 Successfully ingested {chunks} chunks!")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")

def run_query_mode():
    """Run query mode"""
    from config import Config
    from rag_engine import FIRRagEngine
    
    # Get API key
    api_key = input("Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key required for query mode")
        return
    
    Config.set_gemini_api_key(api_key)
    
    try:
        rag = FIRRagEngine(Config)
        
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            
            result = rag.query(query)
            print(f"\n📊 Found {result['retrieved_count']} relevant documents")
            print(f"💬 Answer: {result['answer']}")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")

def main():
    print("🚀 FIR RAG System")
    print("=" * 30)
    print("1. Test System")
    print("2. Ingest Data")
    print("3. Query System")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        # Run comprehensive test
        os.system("python test_system.py")
    elif choice == "2":
        run_ingest_mode()
    elif choice == "3":
        run_query_mode()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()