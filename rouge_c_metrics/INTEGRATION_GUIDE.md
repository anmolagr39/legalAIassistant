# ROUGE-C Integration Guide for IPC RAG System

## Quick Start

### 1. Run Complete Evaluation
```bash
# From project root directory
cd c:\Users\PAVILION\Desktop\legalAIassistant
python rouge_c_metrics\rouge_c_simple.py
```

This will:
- Initialize your IPC RAG system
- Generate Q-A pairs using your existing queries
- Evaluate text generation quality with ROUGE-C metrics
- Save results to `rouge_c_metrics/ipc_rouge_c_results.json`

### 2. Interactive Evaluation
```bash
python rouge_c_metrics\rouge_c_cli.py
```

Choose from:
- Single Q-A pair evaluation
- Batch evaluation from file
- Custom configuration

### 3. Command Line Batch Evaluation
```bash
# Create your Q-A data file first, then:
python rouge_c_metrics\rouge_c_cli.py --batch qa_data.json --output results.json
```

## Integration with Your Existing System

### With Your RAG Engine
```python
# Add to your existing evaluation scripts
import sys
sys.path.append('rouge_c_metrics')
from rouge_c import RougeC

# Initialize
rouge_c = RougeC()

# After getting answer from your RAG system
question = "What is IPC section 302?"
answer = rag_engine.query(question)  # Your existing RAG call
contexts = rag_engine.retriever.query(question, n_results=5)['documents'][0]

# Evaluate
result = rouge_c.compute_rouge_c_single(answer, contexts)
rouge_c.display_results(result)
```

### With Your Existing Evaluation Pipeline
```python
# Add to evaluate_retrieval_metrics.py or similar
from rouge_c_metrics.rouge_c import RougeC

def evaluate_generation_quality():
    rouge_c = RougeC()
    # Use your existing queries from create_ipc_queries.py
    # Generate answers and evaluate
    pass
```

## Expected Output Example

```
🎯 ROUGE-C AGGREGATE RESULTS
=====================================

📊 ROUGE-L METRICS:
   Precision: 0.456 ± 0.123
   Recall:    0.389 ± 0.098
   F1-Score:  0.420 ± 0.107

🔤 TOKEN OVERLAP METRICS:
   Precision: 0.567 ± 0.145
   Recall:    0.432 ± 0.112
   F1-Score:  0.490 ± 0.125

🧠 SEMANTIC SIMILARITY METRICS:
   Combined Context:   0.678 ± 0.087
   Avg Individual:     0.645 ± 0.095
   Max Individual:     0.723 ± 0.078

🎯 PERFORMANCE ASSESSMENT:
   ROUGE-L F1: 0.420 - 🟡 Moderate sequential overlap
   Token F1:   0.490 - 🟡 Moderate lexical overlap  
   Semantic:   0.678 - 🟢 High semantic alignment

🏆 OVERALL SCORE: 0.529
   🟡 GOOD: Answers are reasonably grounded in contexts
```

## Files Generated

After running evaluation:

1. **ipc_qa_data.json** - Generated Q-A pairs for evaluation (project root)
2. **ipc_rouge_c_results.json** - Complete evaluation results (project root)
3. **comprehensive_evaluation_results.json** - Comprehensive results storage (project root)
4. **results_summary_*.json** - Exported summaries (project root)

## Comparison with Retrieval Metrics

Your existing system evaluates **retrieval quality**:
- F1@5: 0.356 (how well documents are retrieved)
- Precision@5: 0.300
- Recall@5: 0.550

ROUGE-C evaluates **generation quality**:
- How well answers use retrieved contexts
- Semantic alignment between answer and contexts
- Lexical overlap and sequential matching

## Performance Interpretation

### Good Performance Indicators
- ROUGE-L F1 > 0.4: Good sequential overlap
- Token F1 > 0.3: Good lexical coverage  
- Semantic Similarity > 0.6: High semantic alignment
- Overall Score > 0.5: Well-grounded answers

### When to Improve
- Low ROUGE-L: Answer structure doesn't match context
- Low Token F1: Answer uses different vocabulary than context
- Low Semantic: Answer semantically divergent from context

## Integration Tips

1. **Run after retrieval evaluation** to get complete pipeline assessment
2. **Use same queries** as your retrieval evaluation for consistency
3. **Monitor trends** as you improve your RAG system
4. **Save results** for comparison across different configurations

## Troubleshooting

### Common Issues
- **Import errors**: Run from project root directory
- **Model download slow**: First run downloads sentence transformer model
- **Memory issues**: Reduce batch size or use lighter embedding model

### Performance Tips
- Use `all-MiniLM-L6-v2` for faster evaluation
- Evaluate in smaller batches for large datasets
- Save intermediate results to avoid re-computation