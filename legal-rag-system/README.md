# Legal RAG System 🏛️

A comprehensive Retrieval-Augmented Generation (RAG) system for Indian Constitution and Legal Acts, powered by ChromaDB, Sentence Transformers, and Google Gemini 2.5 Flash.

## Features ✨

- **Constitution Analysis**: Query articles, fundamental rights, and constitutional provisions
- **Legal Acts Search**: Search through 793K+ legal acts with metadata filtering
- **Hybrid Search**: Intelligent search across both Constitution and Legal Acts
- **Terminal Interface**: Clean, interactive command-line interface
- **Source Attribution**: Detailed citations with similarity scores
- **Batch Processing**: Process multiple queries from files

## Architecture 🏗️

```
Legal RAG System
├── Data Processing
│   ├── PDF Text Extraction (Constitution)
│   ├── CSV Processing (Legal Acts)
│   └── Intelligent Chunking
├── Vector Database (ChromaDB)
│   ├── Constitution Collection
│   └── Legal Acts Collection
├── Embedding Model (SentenceTransformers)
└── LLM (Google Gemini 2.5 Flash)
```

## Installation 🚀

### 1. Prerequisites

- Python 3.8+
- Google Gemini API Key
- At least 4GB RAM (for embeddings)
- 2GB+ storage space

### 2. Setup

```bash
# Navigate to the legal-rag-system directory
cd legal-rag-system

# Install dependencies
python setup.py

# Add your Gemini API key to .env file
# Edit .env file and replace 'your_gemini_api_key_here' with your actual API key

# Initial setup with limited data (for testing)
python main_simple.py --setup --test

# Full setup (processes all 793K legal acts - takes longer)
python main_simple.py --setup
```

### 3. Required Files

Ensure these files are in the parent directory:
- `the_constitution_of_india (1).pdf`
- `legal-acts.csv`

## Usage 💻

### Interactive Mode (Recommended)

```bash
python main_simple.py --interactive
```

Then ask questions like:
- "What are fundamental rights in Indian Constitution?"
- "Explain Article 21 of the Constitution"
- "Show me acts related to consumer protection"
- "What is the right to privacy under Indian law?"

### Single Query Mode

```bash
python main_simple.py --query "What are the fundamental duties of Indian citizens?"
```

### Available Commands

- `--setup`: Setup the system (processes data and creates vector database)
- `--interactive`: Launch interactive query mode
- `--query "question"`: Ask a single question
- `--test`: Use limited data for faster testing
- `--reset`: Reset the vector database
- `--max-acts N`: Limit processing to N legal acts

## System Components 🔧

### 1. Data Processor (`src/data_processor.py`)
- Extracts text from Constitution PDF
- Processes legal acts CSV
- Implements intelligent chunking strategies
- Handles metadata preservation

### 2. Vector Store (`src/vector_store.py`)
- ChromaDB integration
- Embedding generation with SentenceTransformers
- Hybrid search capabilities
- Metadata filtering

### 3. RAG System (`src/rag_system.py`)
- Google Gemini API integration
- Context-aware response generation
- Source attribution and citation
- Legal-specific prompt engineering

## Configuration ⚙️

Key parameters in the system:

```python
CHUNK_SIZE = 1000           # Text chunk size
CHUNK_OVERLAP = 200         # Overlap between chunks
TOP_K = 5                   # Number of retrieved documents
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
```

## Data Statistics 📊

- **Constitution**: ~2.7MB PDF with articles and schedules
- **Legal Acts**: 793,048 acts with full text content
- **Total Storage**: ~2GB after processing
- **Embedding Dimensions**: 384 (MiniLM-L6-v2)

## Example Queries 💡

### Constitutional Questions
```
"What does Article 14 say about equality before law?"
"Explain the concept of basic structure doctrine"
"What are the emergency provisions in the Constitution?"
```

### Legal Acts Questions
```
"Find acts related to information technology"
"What are consumer protection laws in India?"
"Show me recent amendments to criminal law"
```

### Hybrid Questions
```
"How does the Constitution protect personal liberty and what laws implement this?"
"What are the constitutional provisions for freedom of speech and related acts?"
```

## API Integration 🔌

The system is designed to be easily integrated into larger agentic AI workflows:

```python
# Initialize the system
from src.rag_system import LegalRAG
from src.vector_store import VectorStore

vector_store = VectorStore("./data/chroma_db")
rag = LegalRAG(vector_store, "constitution_articles", "legal_acts")

# Query programmatically
result = rag.query("Your legal question here")
print(result['answer'])
print(f"Found {len(result['sources'])} relevant sources")
```

## Performance 📈

- **Setup Time**: 5-30 minutes (depending on data size)
- **Query Time**: 2-5 seconds
- **Memory Usage**: 2-4GB during operation
- **Accuracy**: High relevance with legal document citations

## Troubleshooting 🔧

### Common Issues

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **API Key Issues**: Check your `.env` file and Gemini API key
3. **Memory Issues**: Use `--test` flag for limited data processing
4. **File Not Found**: Ensure PDF and CSV files are in the correct location

### Performance Optimization

- Use SSD storage for better performance
- Increase RAM for faster embedding processing
- Consider using GPU for sentence transformers

## Limitations ⚠️

- Requires internet connection for Gemini API
- Processing large legal acts dataset requires significant time
- Embedding model is English-only
- Responses limited by Gemini's token limits

## Future Enhancements 🚀

- [ ] Multi-language support
- [ ] Real-time legal updates
- [ ] Advanced filtering by jurisdiction
- [ ] Integration with legal precedent databases
- [ ] REST API endpoints
- [ ] Web interface

## Contributing 🤝

This system is part of a larger agentic AI workflow. For integration with other components:

1. Use the programmatic API
2. Follow the established data schemas
3. Maintain compatibility with existing collections

## License 📄

[Add your license information here]

---

**Built for Legal AI Research and Education** 🎓

> This system is designed for educational and research purposes. Always consult qualified legal professionals for legal advice.