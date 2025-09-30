# FIR Legal AI Assistant

A sophisticated RAG (Retrieval-Augmented Generation) system for analyzing Indian FIR (First Information Report) data using ChromaDB vector database and Google Gemini AI.

## 🚀 Features

- **📊 FIR Dataset Processing**: Intelligent ingestion and preprocessing of FIR datasets
- **🔍 Semantic Search**: Advanced semantic search using sentence transformers
- **🤖 AI-Powered Analysis**: Google Gemini AI integration for legal analysis
- **📋 Fallback Analysis**: Rule-based fallback system for quota management
- **⚖️ IPC Section Extraction**: Automatic identification and analysis of Indian Penal Code sections
- **💾 Vector Database**: ChromaDB for efficient document storage and retrieval
- **🎯 CLI Interface**: Easy-to-use command-line interface

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Setup
```bash
# Clone the repository
git clone https://github.com/anmolagr39/legalAIassistant.git
cd legalAIassistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration

1. Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Set up your API key (you'll provide it as a command-line argument)

## 📖 Usage

### Data Ingestion
```bash
# Ingest FIR dataset into the vector database
python main.py --mode ingest --csv "path/to/your/FIR_DATASET.csv"
```

### Querying the System
```bash
# Query the RAG system
python main.py --mode query --query "What are the IPC sections for cheating?" --api-key "your_gemini_api_key"

# Example queries:
python main.py --mode query --query "Cases related to fraud" --api-key "your_api_key"
python main.py --mode query --query "IPC sections for theft" --api-key "your_api_key"
```

### Alternative Runner
```bash
# Use the alternative runner script
python run_rag.py
```

## 📁 Project Structure

```
legalAIassistant/
├── src/
│   ├── config.py          # Configuration management and utilities
│   └── rag_engine.py      # Core RAG engine implementation
├── main.py                # Main CLI application entry point
├── run_rag.py            # Alternative runner script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore rules
```

## 🧩 Architecture

### Core Components

1. **Configuration Manager** (`src/config.py`)
   - API key management
   - Path configuration
   - Utility functions

2. **RAG Engine** (`src/rag_engine.py`)
   - ChromaDB integration
   - Google Gemini AI integration
   - Document preprocessing and chunking
   - Semantic search and retrieval
   - Fallback analysis system

3. **CLI Interface** (`main.py`)
   - Argument parsing
   - Mode switching (ingest/query)
   - User interaction

### Data Flow

1. **Ingestion**: FIR data → Preprocessing → Chunking → Embedding → ChromaDB
2. **Query**: User query → Embedding → Similarity search → Context retrieval → Gemini AI → Response

## 🔍 Features in Detail

### Intelligent Text Processing
- Automatic text column detection
- Smart chunking for optimal retrieval
- IPC section extraction using regex patterns

### AI Integration
- Google Gemini 1.5 Flash model for analysis
- Quota management with graceful fallbacks
- Context-aware legal analysis

### Fallback System
- Rule-based analysis when AI quota is exceeded
- IPC section identification and explanation
- Maintains system availability

## 🎯 Example Use Cases

- **Legal Research**: Find relevant FIR cases based on keywords
- **IPC Analysis**: Identify applicable Indian Penal Code sections
- **Case Similarity**: Find similar cases for legal precedent
- **Pattern Analysis**: Discover patterns in criminal cases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Legal analysis should always be verified by qualified legal professionals.

## 🙏 Acknowledgments

- Google Gemini AI for powerful language model capabilities
- ChromaDB for efficient vector database operations
- Sentence Transformers for semantic embeddings
- The open-source community for various libraries and tools

---

**Author**: rakshita25  
**Project**: Legal AI Assistant for FIR Analysis  
**Version**: 1.0.0