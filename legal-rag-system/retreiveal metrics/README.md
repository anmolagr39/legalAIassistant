# Retrieval Metrics Module

This module implements retrieval-based evaluation metrics for RAG systems without requiring ground truth annotations. It uses semantic similarity and consistency-based approaches following recent research in retrieval evaluation.

## Features

### Metrics Implemented

1. **Semantic Precision**: Measures the proportion of retrieved documents that are semantically relevant to the query based on embedding similarity.

2. **Estimated Recall**: Estimates recall by comparing retrieved documents against a larger corpus sample to identify potentially relevant documents.

3. **F1-Score**: Harmonic mean of precision and recall for balanced evaluation.

4. **Document Coherence**: Measures consistency among retrieved documents (high coherence indicates focused retrieval).

5. **Retrieval Diversity**: Measures variety in retrieved content (complement of coherence).

6. **Average Query Similarity**: Mean semantic similarity between query and retrieved documents.

## Usage

### Command Line Interface

Run the interactive CLI to test your RAG system:

```bash
cd "retreiveal metrics"
python metrics_cli.py
```

The CLI provides:
- Single query evaluation with immediate metrics
- Batch evaluation with predefined legal queries
- System status and statistics
- Results export to JSON

### Programmatic Usage

```python
from retrieval_metrics import RetrievalMetrics

# Initialize evaluator
evaluator = RetrievalMetrics(
    embedding_model="all-MiniLM-L6-v2",
    similarity_threshold=0.7
)

# Evaluate single query
metrics = evaluator.evaluate_retrieval(
    query="What are fundamental rights?",
    retrieved_docs=["doc1", "doc2", "doc3"],
    all_available_docs=full_corpus  # optional for recall
)

# Batch evaluation
results = evaluator.batch_evaluate(query_results)
evaluator.display_results(results)
```

## Configuration

- **Similarity Threshold**: Minimum cosine similarity to consider a document relevant (default: 0.7)
- **Embedding Model**: Sentence transformer model for semantic similarity (default: "all-MiniLM-L6-v2")
- **Top-K for Recall**: Number of most similar documents in corpus to use for recall estimation (default: 20)

## Interpretation

### Metric Ranges
- **Precision**: [0, 1] - Higher is better
- **Recall**: [0, 1] - Higher is better  
- **F1-Score**: [0, 1] - Higher is better
- **Coherence**: [0, 1] - Higher indicates more focused retrieval
- **Diversity**: [0, 1] - Higher indicates more varied content
- **Avg Query Similarity**: [-1, 1] - Higher indicates better query-document alignment

### Good Performance Indicators
- Precision > 0.7: Most retrieved documents are relevant
- Recall > 0.5: Good coverage of relevant content
- F1-Score > 0.6: Balanced precision and recall
- Coherence 0.4-0.8: Focused but not redundant retrieval
- Diversity 0.2-0.6: Some variety without losing focus

## Files

- `retrieval_metrics.py`: Core metrics implementation
- `metrics_cli.py`: Interactive command-line interface
- `README.md`: This documentation

## Dependencies

The module uses existing project dependencies:
- sentence-transformers
- scikit-learn (for cosine similarity)
- numpy
- rich (for CLI formatting)
- pandas (for data handling)