# Knowledge Graph Evaluation Metrics

This folder contains evaluation metrics for the Knowledge Graph system, adapted from the legal-rag-system metrics.

## Metrics Implemented

### 1. Retrieval Metrics (`kg_metrics.py`)
- **Precision**: Measures relevance of retrieved KG results based on semantic similarity
- **Recall**: Estimates completeness of retrieved results
- **F1 Score**: Harmonic mean of precision and recall
- **Coherence**: Measures consistency among retrieved results
- **Query Similarity**: Average semantic similarity between query and results

### 2. Generation Metrics (`kg_generation_metrics.py`)
- **ROUGE-L**: Longest Common Subsequence based metric for text quality
- **LCS Score**: Normalized Longest Common Subsequence score
- **SUPERT-style Metrics**: Sentence-level semantic similarity using optimal matching
  - SUPERT F1, Precision, Recall
  - Mean sentence similarity
  - Number of matched sentence pairs

## Files

- `kg_metrics.py` - Retrieval-based metrics (Precision, Recall, F1)
- `kg_generation_metrics.py` - Generation quality metrics (ROUGE-L, LCS, SUPERT)
- `run_kg_evaluation.py` - Main evaluation runner script
- `README.md` - This file

## Usage

### Run Complete Evaluation

```bash
cd D:\legalkg\metrics\kgmetrics
python run_kg_evaluation.py
```

This will:
1. Execute test queries against the Knowledge Graph
2. Compute retrieval metrics (precision, recall, F1)
3. Compute generation metrics (ROUGE-L, LCS, SUPERT)
4. Display results in formatted tables
5. Save results to JSON and CSV files

### Output Files

- `kg_retrieval_metrics.json` - Retrieval metric results
- `kg_generation_metrics.json` - Generation metric results
- `kg_evaluation_results.csv` - Detailed per-query results

## Test Queries

The evaluation uses **15 carefully selected queries** that work well with the available Knowledge Graph data:

### IPC Section Queries (10 queries)
- Specific section lookups (302, 307, 376, 420, 498A, 354)
- Filtered queries (bailable, non-bailable, cognizable offenses)
- Punishment queries

### Judge & Case Queries (5 queries)
- Judge statistics (most cases, >10 cases)
- Cases by judge
- Cases involving specific IPC sections
- General case searches

These queries were selected because:
- They match the actual data in the KG (445 IPC sections, 1482 cases, 189 judges)
- They exercise different query patterns (exact match, filtering, aggregation)
- They represent realistic user queries
- They have high success rates based on testing

## Metrics Explanation

### Retrieval Metrics

**Precision**: What proportion of retrieved results are relevant?
- Computed using semantic similarity threshold (default 0.7)
- Higher is better (0-1 range)

**Recall**: What proportion of relevant results were retrieved?
- Estimated based on result quality for KG
- Higher is better (0-1 range)

**F1 Score**: Balanced measure of precision and recall
- Harmonic mean: `2 * (P * R) / (P + R)`
- Higher is better (0-1 range)

### Generation Metrics

**ROUGE-L**: Measures text similarity using Longest Common Subsequence
- Precision, Recall, F1 computed at token level
- Higher is better (0-1 range)

**LCS Score**: Normalized longest common subsequence
- Normalized by average text length
- Higher is better (0-1 range)

**SUPERT F1**: Sentence-level semantic similarity
- Uses optimal bipartite matching of sentences
- Based on sentence transformer embeddings
- Higher is better (0-1 range)

## Requirements

- sentence-transformers
- scikit-learn
- nltk
- neo4j
- pandas
- rich
- numpy
- scipy

All requirements should already be installed in the main virtual environment.

## Connection Details

- **Neo4j URI**: bolt://localhost:7687
- **Database**: neo4j
- **Auth**: neo4j/anmol1234

Make sure Neo4j is running before executing the evaluation.

## Interpretation

### Good Performance Indicators
- Precision > 0.7: Most results are relevant
- F1 > 0.6: Good balance of precision and recall
- ROUGE-L F1 > 0.5: Generated text matches KG context well
- SUPERT F1 > 0.6: Semantic similarity between answer and context is high

### Areas for Improvement
- Precision < 0.5: Too many irrelevant results
- Recall < 0.5: Missing relevant results
- ROUGE-L F1 < 0.3: Generated text doesn't match context
- SUPERT F1 < 0.4: Weak semantic alignment

## Comparison with RAG System

These metrics are adapted from the legal-rag-system but tailored for Knowledge Graph evaluation:

1. **Retrieval**: KG retrieval is more structured (Cypher queries) vs RAG's semantic search
2. **Generation**: KG answers are synthesized from structured data vs RAG's document chunks
3. **Context**: KG context is query results vs RAG's retrieved documents
4. **Evaluation Focus**: KG metrics emphasize query accuracy and result relevance vs RAG's document retrieval quality

## Notes

- The evaluation uses a simplified answer generation approach (not the full LLM synthesis from the orchestrator)
- Metrics are computed using semantic similarity and sentence transformers
- Results depend on the quality of the Cypher query generation and KG data completeness
- Some queries may fail if data is not available in the KG
