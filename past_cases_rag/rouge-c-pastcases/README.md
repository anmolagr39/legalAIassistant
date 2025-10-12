# ROUGE-C: Reference-Free Context-Based ROUGE Metrics for Legal Past Cases RAG

This module implements ROUGE-C, a reference-free variant of ROUGE metrics that evaluates generated text quality against retrieved legal case contexts without requiring reference answers. It's specifically tailored for the Legal Past Cases RAG system.

## Overview

ROUGE-C adapts traditional ROUGE metrics for reference-free evaluation by comparing generated answers against retrieved legal case contexts instead of gold-standard references. This makes it ideal for legal RAG systems where reference answers may not be available.

## Metrics Implemented

### 1. ROUGE-L (Longest Common Subsequence)
- **Description**: Measures sequential overlap between generated answer and legal case contexts
- **Computation**: Uses dynamic programming to find longest common subsequence of tokens
- **Output**: Precision, Recall, and F1-Score based on LCS length
- **Good Range**: F1 > 0.5 indicates strong sequential alignment with legal cases

### 2. Token Overlap 
- **Description**: Measures lexical overlap between answer and legal case context tokens
- **Computation**: Set intersection of preprocessed tokens (preserves legal terminology)
- **Output**: Precision, Recall, and F1-Score based on token overlap
- **Good Range**: F1 > 0.4 indicates strong lexical alignment with legal content

### 3. Legal Term Overlap
- **Description**: Specialized metric measuring overlap of legal terminology
- **Computation**: Focuses on legal-specific terms (constitutional, judgment, petition, etc.)
- **Output**: Precision, Recall, and F1-Score for legal terminology alignment
- **Good Range**: F1 > 0.3 indicates good use of legal vocabulary

### 4. Semantic Similarity
- **Description**: Measures semantic alignment using embedding-based similarity
- **Computation**: Cosine similarity between sentence embeddings of answer and contexts
- **Output**: Similarity scores for combined context and individual contexts
- **Good Range**: Similarity > 0.7 indicates high semantic alignment with legal cases

## Features

### Legal-Specific Text Preprocessing
- Tokenization and normalization optimized for legal text
- Preserves important legal terms while removing common stopwords
- Optional stemming (requires NLTK)
- Case normalization with legal term awareness

### Legal Term Recognition
- Specialized vocabulary for constitutional law, criminal law, civil law
- Weighted importance for legal terminology (constitutional, amendment, petition, etc.)
- Domain-specific preprocessing that maintains legal context

### Evaluation Modes
- **Single Evaluation**: Evaluate one legal question-answer pair
- **Batch Evaluation**: Evaluate multiple pairs with aggregate statistics
- **Context Combination**: Handles multiple retrieved legal case contexts per query

### Output Metrics
- Individual metrics for each legal Q-A pair
- Aggregate statistics (mean, std deviation)
- Legal domain performance interpretation
- Overall ROUGE-C score combining all metrics

## Files Structure

```
rouge-c-pastcases/
├── rouge_c_pastcases.py          # Main ROUGE-C implementation for legal cases
├── legal_queries_pastcases.py    # Legal queries for evaluation
├── rouge_c_cli.py               # Simple CLI evaluation script
├── run_rouge_c_evaluation.py    # Full evaluation script (requires rich)
└── README.md                    # This file
```

## Usage

### Quick Evaluation (Recommended)
Run the simple CLI evaluation:
```bash
cd rouge-c-pastcases
python rouge_c_cli.py
```

This will:
- Initialize the Legal Past Cases RAG system
- Run ROUGE-C evaluation on 10 legal queries
- Display individual and aggregate results
- Save results to JSON file

### Advanced Evaluation (Requires rich library)
For enhanced output formatting:
```bash
python run_rouge_c_evaluation.py
```

### Programmatic Usage
```python
from rouge_c_pastcases import LegalPastCasesRougeC
from legal_queries_pastcases import get_simple_legal_queries

# Initialize evaluator
evaluator = LegalPastCasesRougeC()

# Evaluate single question-answer pair
question = "What are the key Supreme Court judgments on Article 21?"
answer = "Article 21 protects right to life and personal liberty..."
contexts = ["Retrieved legal case 1...", "Retrieved legal case 2..."]

result = evaluator.evaluate_single(question, answer, contexts)
evaluator.display_results(result)
```

## Configuration

### LegalPastCasesRougeC Parameters
- `embedding_model`: Model for semantic similarity (default: "all-MiniLM-L6-v2")
- `use_stemming`: Whether to apply stemming (default: False)
- `remove_stopwords`: Whether to remove stopwords while preserving legal terms (default: True)

### Legal Term Categories
The system recognizes these legal term categories:
- **Constitutional**: constitutional, amendment, article, fundamental, rights
- **Procedural**: petition, writ, mandamus, certiorari, habeas, corpus
- **Judicial**: supreme, high, tribunal, judgment, appeal, criminal, civil
- **Administrative**: judicial, review, administrative, jurisdiction, precedent

## Interpretation Guide

### Overall ROUGE-C Score Ranges
- **0.7+ (Excellent)**: Strong alignment between answers and legal case contexts
- **0.5-0.7 (Good)**: Moderate alignment, answers utilize retrieved cases well
- **0.3-0.5 (Fair)**: Some alignment, but improvement needed in context utilization
- **<0.3 (Poor)**: Weak alignment, significant improvement needed

### Individual Metric Interpretation
- **ROUGE-L F1 > 0.4**: Good sequential overlap with legal case content
- **Token Overlap F1 > 0.4**: Strong lexical similarity with case contexts
- **Legal Term F1 > 0.3**: Appropriate use of legal terminology from cases
- **Semantic Similarity > 0.7**: High semantic alignment with case meanings

## Example Output

```
Query 1: What are the key Supreme Court judgments on right to life under Article 21?
Retrieved contexts: 5
Generated answer: Article 21 of the Constitution protects the right to life...

ROUGE-C Metrics:
  ROUGE-L     - P: 0.421, R: 0.387, F1: 0.403
  Token Overlap - P: 0.456, R: 0.392, F1: 0.422
  Legal Terms - P: 0.667, R: 0.400, F1: 0.500
  Semantic Similarity - Combined: 0.743, Max: 0.801, Avg: 0.732
  Overall Score: 0.517

BATCH STATISTICS
Number of evaluations: 10
ROUGE-L F1        - Mean: 0.387 ± 0.092
Token Overlap F1  - Mean: 0.401 ± 0.078
Legal Terms F1    - Mean: 0.334 ± 0.124
Semantic Sim      - Mean: 0.681 ± 0.087
Overall Score     - Mean: 0.451 ± 0.071

✓ System shows moderate alignment - consider improving answer generation
```

## Dependencies

- sentence-transformers: For semantic similarity computation
- scikit-learn: For cosine similarity calculations
- numpy: For numerical operations
- rich (optional): For enhanced console output
- nltk (optional): For stemming functionality

## Integration with Legal RAG System

The ROUGE-C evaluator integrates with your existing Legal Past Cases RAG system:
- Uses the same ChromaVectorDB for context retrieval
- Works with LegalRAGSystem for query processing
- Evaluates the quality of answer-context alignment
- Provides feedback for system optimization

## Performance Optimization

For better ROUGE-C scores:
1. **Improve Context Retrieval**: Ensure relevant legal cases are retrieved
2. **Enhance Answer Generation**: Generate answers that better utilize retrieved contexts
3. **Legal Term Usage**: Ensure generated answers use appropriate legal terminology
4. **Semantic Alignment**: Improve semantic similarity between answers and case contexts