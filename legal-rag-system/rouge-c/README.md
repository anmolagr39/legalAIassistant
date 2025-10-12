# ROUGE-C: Reference-Free Context-Based ROUGE Metrics

This module implements ROUGE-C, a reference-free variant of ROUGE metrics that evaluates generated text quality against retrieved contexts without requiring reference answers. It's specifically designed for RAG (Retrieval-Augmented Generation) systems.

## Overview

ROUGE-C adapts traditional ROUGE metrics for reference-free evaluation by comparing generated answers against retrieved contexts instead of gold-standard references. This makes it ideal for RAG systems where reference answers may not be available.

## Metrics Implemented

### 1. ROUGE-L (Longest Common Subsequence)
- **Description**: Measures sequential overlap between generated answer and context
- **Computation**: Uses dynamic programming to find longest common subsequence of tokens
- **Output**: Precision, Recall, and F1-Score based on LCS length
- **Good Range**: F1 > 0.5 indicates strong sequential alignment

### 2. Token Overlap 
- **Description**: Measures lexical overlap between answer and context tokens
- **Computation**: Set intersection of preprocessed tokens
- **Output**: Precision, Recall, and F1-Score based on token overlap
- **Good Range**: F1 > 0.4 indicates strong lexical alignment

### 3. Semantic Similarity
- **Description**: Measures semantic alignment using embedding-based similarity
- **Computation**: Cosine similarity between sentence embeddings
- **Output**: Similarity scores for combined context and individual contexts
- **Good Range**: Similarity > 0.7 indicates high semantic alignment

## Features

### Text Preprocessing
- Tokenization and normalization
- Optional stopword removal (enabled by default)
- Optional stemming (requires NLTK)
- Case normalization

### Evaluation Modes
- **Single Evaluation**: Evaluate one question-answer pair
- **Batch Evaluation**: Evaluate multiple pairs with aggregate statistics
- **Context Combination**: Handles multiple retrieved contexts per query

### Output Metrics
- Individual metrics for each Q-A pair
- Aggregate statistics (mean, std deviation)
- Performance interpretation and scoring

## Usage

### Quick Evaluation
Run automatic evaluation on 10 legal queries:
```bash
cd rouge-c
python run_rouge_c_evaluation.py
```

### Interactive CLI
```bash
cd rouge-c
python rouge_c_cli.py
```

### Programmatic Usage
```python
from rouge_c import RougeC

# Initialize evaluator
evaluator = RougeC(
    embedding_model="all-MiniLM-L6-v2",
    remove_stopwords=True,
    use_stemming=False
)

# Single evaluation
result = evaluator.compute_rouge_c_single(
    generated_text="The fundamental rights include...",
    retrieved_contexts=["Article 12 defines...", "Article 13 states..."]
)

# Batch evaluation
qa_pairs = [
    {
        'question': 'What are fundamental rights?',
        'generated_answer': 'Fundamental rights are...',
        'retrieved_contexts': ['Context 1', 'Context 2']
    }
]

results = evaluator.compute_rouge_c_batch(qa_pairs)
evaluator.display_results(results)
```

## Configuration Options

### Embedding Model
- Default: `"all-MiniLM-L6-v2"`
- Other options: Any sentence-transformers model
- Impact: Affects semantic similarity computation

### Text Preprocessing
- **remove_stopwords**: Remove common words (default: True)
- **use_stemming**: Apply Porter stemming (default: False, requires NLTK)

### Stopwords
Built-in English stopword list including articles, prepositions, pronouns, and common verbs.

## Output Interpretation

### Score Ranges
- **ROUGE-L F1**: [0, 1] - Higher indicates better sequential overlap
- **Token Overlap F1**: [0, 1] - Higher indicates better lexical overlap  
- **Semantic Similarity**: [0, 1] - Higher indicates better semantic alignment

### Performance Levels
- **Excellent (>0.6)**: Well-grounded answers with strong context alignment
- **Good (0.4-0.6)**: Reasonably grounded answers with adequate alignment
- **Fair (0.2-0.4)**: Some grounding in contexts but room for improvement
- **Poor (<0.2)**: Poorly grounded answers with weak context alignment

### Example Output
```
ROUGE-C Metrics Summary
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Precision     ┃ Recall      ┃ F1-Score      ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ ROUGE-L       │ 0.654 ±0.120  │ 0.432 ±0.089│ 0.521 ±0.098  │
│ Token Overlap │ 0.712 ±0.145  │ 0.389 ±0.102│ 0.504 ±0.112  │
└───────────────┴───────────────┴─────────────┴───────────────┘

Semantic Similarity: 0.731 ±0.087
Overall Score: 0.585 (Good)
```

## Files

- `rouge_c.py`: Core ROUGE-C implementation
- `rouge_c_cli.py`: Interactive command-line interface
- `run_rouge_c_evaluation.py`: Automatic evaluation script
- `README.md`: This documentation

## Dependencies

- sentence-transformers (for embeddings)
- scikit-learn (for cosine similarity)
- numpy (for numerical operations)
- rich (for formatted output)
- nltk (optional, for stemming)

## Technical Details

### ROUGE-L Algorithm
Uses dynamic programming to compute longest common subsequence:
```
LCS(seq1, seq2) = max length of common subsequence
Precision = LCS_length / len(generated_tokens)
Recall = LCS_length / len(context_tokens)  
F1 = 2 * Precision * Recall / (Precision + Recall)
```

### Token Overlap Algorithm
```
overlap = intersection(generated_tokens, context_tokens)
Precision = |overlap| / |generated_tokens|
Recall = |overlap| / |context_tokens|
F1 = 2 * Precision * Recall / (Precision + Recall)
```

### Semantic Similarity Algorithm
```
gen_embedding = encode(generated_text)
ctx_embedding = encode(context_text)  
similarity = cosine_similarity(gen_embedding, ctx_embedding)
normalized_sim = (similarity + 1) / 2  # Map [-1,1] to [0,1]
```

## References

1. Lin, C.Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries". ACL 2004.
2. Adapted for reference-free RAG evaluation following context-grounding principles.
3. Semantic similarity approach inspired by modern embedding-based evaluation methods.