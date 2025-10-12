# Legal Assistant RAG System

A Retrieval-Augmented Generation system for Indian Supreme Court case documents using ChromaDB and Gemini 2.5 Flash.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Initialize the database:
```bash
python main.py --init
```

5. Ingest documents:
```bash
python main.py --ingest
```

6. Query the system:
```bash
python main.py --query "What are the landmark cases on fundamental rights?"
```

## Usage

```bash
python main.py --help
```

## Dataset
Contains 2,914 Indian Supreme Court case documents in text format.

## Architecture
- **Document Processing**: Extract and chunk legal documents
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: Sentence transformers for text embeddings
- **LLM**: Gemini 2.5 Flash for response generation
- **Interface**: Command-line interface for testing