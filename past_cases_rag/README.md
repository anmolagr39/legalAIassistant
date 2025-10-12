# Past Cases RAG System

This directory contains the Retrieval-Augmented Generation (RAG) system for legal case analysis.

## Structure

- `src/` - Source code for RAG processing, utilities, and main logic
- `tests/` - Test files for the RAG system  
- `Object_casedocs/` - Legal case documents (not included in version control)
- `chroma_db/` - Vector database files (not included in version control)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Add your case documents to the `Object_casedocs/` directory
3. Run the ingestion process to build the vector database
4. Use the RAG system to query legal cases

## Data Files

Note: Large data files (case documents and vector databases) are excluded from version control for size reasons.