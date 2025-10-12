# ROUGE-C Metrics for IPC RAG System

Reference-free evaluation metrics for measuring the quality of generated text against retrieved context without requiring reference answers.

## Overview

ROUGE-C (Context-based ROUGE) evaluates how well generated answers are grounded in retrieved contexts using three complementary metrics:

1. **ROUGE-L**: Longest Common Subsequence overlap between generated text and contexts
2. **Token Overlap**: Precision, recall, and F1 of token matches  
3. **Semantic Similarity**: Cosine similarity between embeddings using sentence transformers

## Features

- 🎯 **Reference-free evaluation** - No gold standard answers needed
- 📊 **Multiple metrics** - ROUGE-L, token overlap, and semantic similarity
- 🔤 **Text preprocessing** - Stopword removal, optional stemming
- 📈 **Batch evaluation** - Evaluate multiple Q-A pairs efficiently
- 💻 **CLI interface** - Interactive command-line evaluation
- 🚀 **Simple automation** - One-line evaluation for IPC RAG system
- 💾 **Result persistence** - Save evaluation data and results to JSON

## Installation

The package requires the following dependencies (already in your project's `requirements.txt`):

```bash
pip install sentence-transformers scikit-learn numpy
```

Optional for stemming:
```bash
pip install nltk
```

## Quick Start

### 1. Simple Automated Evaluation

```python
from rouge_c_metrics import run_simple_evaluation

# Run automated evaluation on IPC RAG system
results = run_simple_evaluation()
```

### 2. Direct Evaluation

```python
from rouge_c_metrics import RougeC

# Initialize evaluator
rouge_c = RougeC()

# Evaluate single Q-A pair
question = "What is IPC section 302?"
generated_answer = "IPC section 302 deals with punishment for murder..."
retrieved_contexts = ["Section 302 of Indian Penal Code...", "Murder provisions..."]

result = rouge_c.compute_rouge_c_single(generated_answer, retrieved_contexts)
rouge_c.display_results(result)
```

### 3. Batch Evaluation

```python
from rouge_c_metrics import RougeC

qa_pairs = [
    {
        "question": "What is IPC section 302?",
        "generated_answer": "IPC section 302 deals with murder...",
        "retrieved_contexts": ["Section 302 text...", "Murder definition..."]
    },
    # ... more pairs
]

rouge_c = RougeC()
results = rouge_c.compute_rouge_c_batch(qa_pairs)
rouge_c.display_results(results)
```

### 4. Interactive CLI

```python
from rouge_c_metrics import RougeCCLI

cli = RougeCCLI()
cli.interactive_evaluation()
```

Or from command line:
```bash
python rouge_c_cli.py
python rouge_c_cli.py --batch qa_data.json --output results.json
```

## Usage Examples

### Command Line Interface

```bash
# Interactive mode
python rouge_c_cli.py

# Batch evaluation
python rouge_c_cli.py --batch qa_data.json --output results.json

# Single evaluation
python rouge_c_cli.py --question "What is IPC 302?" --answer "Murder section" --contexts "Section 302 deals with murder"

# Custom configuration
python rouge_c_cli.py --batch qa_data.json --model all-mpnet-base-v2 --stemming --output results.json
```

### Automated Testing

```bash
# Test implementation
python test_rouge_c.py

# Run full evaluation on IPC RAG system
python rouge_c_simple.py
```

## File Structure

```
rouge_c_metrics/
├── __init__.py              # Package initialization
├── rouge_c.py               # Core ROUGE-C implementation
├── rouge_c_cli.py           # Command-line interface
├── rouge_c_simple.py        # Simple evaluation for IPC RAG
├── test_rouge_c.py          # Test script
├── README.md                # This file
├── ipc_qa_data.json         # Generated Q-A pairs (created during evaluation)
└── ipc_rouge_c_results.json # Evaluation results (created during evaluation)
```

## Configuration Options

### Embedding Models

- `all-MiniLM-L6-v2` (default) - Fast, good performance
- `all-mpnet-base-v2` - More accurate, slower
- Any sentence-transformers model

### Text Processing

- `remove_stopwords` (default: True) - Remove common stopwords
- `use_stemming` (default: False) - Apply Porter stemming (requires nltk)

### Evaluation Parameters

- `num_contexts` - Number of contexts to retrieve (default: 5)
- `save_qa_data` - Save generated Q-A pairs (default: True)
- `save_results` - Save evaluation results (default: True)

## Metrics Interpretation

### ROUGE-L Scores
- **Precision**: % of generated tokens in longest common subsequence
- **Recall**: % of context tokens in longest common subsequence  
- **F1**: Harmonic mean of precision and recall
- **Range**: 0.0 - 1.0 (higher is better)

### Token Overlap Scores
- **Precision**: % of generated tokens found in contexts
- **Recall**: % of context tokens found in generated text
- **F1**: Harmonic mean of precision and recall
- **Range**: 0.0 - 1.0 (higher is better)

### Semantic Similarity
- **Score**: Cosine similarity between embeddings
- **Range**: 0.0 - 1.0 (higher is better)
- **Interpretation**: 
  - 0.7+ = High semantic alignment
  - 0.5-0.7 = Moderate alignment  
  - <0.5 = Low alignment

### Overall Performance Assessment

The system provides an overall score by averaging the three F1 metrics:

- **0.6+**: 🟢 Excellent - Answers well-grounded in contexts
- **0.4-0.6**: 🟡 Good - Reasonably grounded answers
- **0.2-0.4**: 🟠 Fair - Some grounding in contexts
- **<0.2**: 🔴 Poor - Poorly grounded answers

## Data Format

### Q-A Pairs Format (JSON)

```json
[
    {
        "question": "What is IPC section 302?",
        "generated_answer": "IPC section 302 deals with punishment for murder...",
        "retrieved_contexts": [
            "Section 302 of Indian Penal Code: Punishment for murder...",
            "Murder provisions under IPC..."
        ]
    }
]
```

### Results Format

```json
{
    "aggregate_metrics": {
        "rouge_l": {
            "avg_f1": 0.456,
            "std_f1": 0.123,
            "avg_precision": 0.389,
            "avg_recall": 0.567
        },
        "token_overlap": {
            "avg_f1": 0.345,
            "std_f1": 0.098
        },
        "semantic_similarity": {
            "avg_combined": 0.678,
            "std_combined": 0.087
        }
    },
    "individual_results": [...]
}
```

## Integration with IPC RAG System

The package integrates seamlessly with your existing IPC RAG system:

1. **Automatic Q-A Generation**: Uses your existing queries from `create_ipc_queries.py`
2. **RAG Engine Integration**: Leverages `src/rag_engine.py` for answer generation
3. **Context Retrieval**: Uses the same ChromaDB retrieval as your retrieval metrics
4. **Consistent Evaluation**: Complements your existing retrieval evaluation metrics

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Model download**: First run may take time to download embedding models
3. **Memory issues**: Use lighter embedding models for large datasets
4. **Empty results**: Check that Q-A pairs have non-empty text

### Performance Tips

1. Use `all-MiniLM-L6-v2` for faster evaluation
2. Disable stemming for better performance
3. Reduce `num_contexts` for faster processing
4. Use batch evaluation for multiple queries

## Contributing

This implementation is adapted for the IPC RAG system. Key adaptations:

- Legal domain terminology preservation
- Integration with existing RAG pipeline
- Optimized for Indian legal document evaluation
- Compatible with existing evaluation framework

## References

- Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL.
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- ChromaDB: Vector database for semantic search