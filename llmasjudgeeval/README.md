# LLM-as-a-Judge Evaluation System

This folder contains the LLM-as-a-judge evaluation system for testing the Legal Assistant's agentic orchestrator against ground truth data.

## Files

### Core Files
- **`ground_truth_dataset.json`** - 30 ground truth questions with answers based on ingested data from 3 RAGs and Knowledge Graph
- **`llm_judge_evaluator.py`** - Main evaluation system with LLM judge implementation
- **`run_evaluation.py`** - Convenient scripts for running evaluations (quick tests, category-specific)
- **`analyze_results.py`** - Analysis tools for evaluation results
- **`read_ingested_data.py`** - Script to read data from ChromaDB and Neo4j (used for dataset creation)

### Output
- **`evaluation_results/`** - Directory containing evaluation results, reports, and analysis

## Ground Truth Dataset

The dataset contains 30 questions covering:

- **IPC RAG** (8 questions) - IPC Sections 127-134, 140
- **Constitution RAG** (7 questions) - Supreme Court cases on Articles 12, 14, 21
- **Past Cases RAG** (15 questions) - Cases from 1973, 1987, 2001 covering various legal principles

Each question includes:
- Question text
- Ground truth answer
- Category
- Data source

## Usage

### 1. Run Full Evaluation (All 30 Questions)

```bash
python llm_judge_evaluator.py
```

This will:
- Query the agentic orchestrator for each question
- Evaluate responses using Gemini as LLM judge
- Generate detailed JSON results
- Create summary reports

### 2. Quick Test (5 Questions)

```bash
python run_evaluation.py --quick 5
```

### 3. Category-Specific Evaluation

List available categories:
```bash
python run_evaluation.py --list-categories
```

Evaluate specific category:
```bash
python run_evaluation.py --category IPC_Section
python run_evaluation.py --category Constitutional_Law
python run_evaluation.py --category Negotiable_Instruments
```

### 4. Analyze Results

```bash
python analyze_results.py evaluation_results/final_results_TIMESTAMP.json
```

This generates:
- Detailed analysis report
- Category breakdown
- Top/bottom performers
- Common weaknesses
- Failed questions details

## Evaluation Metrics

The LLM judge evaluates each response on 4 dimensions (0-10 scale):

1. **Correctness** - Accuracy of legal facts, case names, section numbers
2. **Completeness** - Coverage of all key points from ground truth
3. **Relevance** - Direct addressing of the question
4. **Legal Reasoning** - Quality of legal reasoning and citations

**Overall Score**: Average of the 4 metrics

**Pass Criteria**: Overall score ≥ 6.0

## Output Files

After running evaluation, check `evaluation_results/` for:

- `intermediate_results_TIMESTAMP.json` - Saved after each question
- `evaluation_report.json` - Final complete results with statistics
- `summary_report_TIMESTAMP.txt` - Human-readable summary
- `detailed_analysis_report.txt` - Detailed analysis (after running analyze_results.py)

## Configuration

Edit these parameters in the scripts:

- `delay_seconds` - Delay between API calls (default: 2.0s)
- `judge_model` - Gemini model for judging (default: gemini-2.0-flash-exp)
- `max_questions` - Limit number of questions (None = all)

## Requirements

Ensure these are installed:
```bash
pip install google-generativeai python-dotenv chromadb neo4j
```

Required environment variables in `.env`:
```
GEMINI_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Example Output

```
[1/30] Evaluating Question 1
Category: IPC_Section
Question: What is the punishment for wearing a soldier's uniform...
  → Querying orchestrator...
  → Got response in 3.45s
  → Systems used: knowledge_graph, ipc_rag
  → Evaluating with LLM judge...
  → Evaluation complete in 2.31s
  → Verdict: PASS (Overall Score: 8.5/10)
  ✓ Question 1 complete

...

Evaluation Complete!
================================================================================

Total Questions: 30
Passed: 24 (80.0%)
Failed: 6 (20.0%)
Average Overall Score: 7.23/10

Results saved to: evaluation_results/
```

## Troubleshooting

**Issue**: Rate limit errors
- **Solution**: Increase `delay_seconds` parameter

**Issue**: Neo4j connection failed
- **Solution**: Ensure Neo4j is running and credentials are correct in `.env`

**Issue**: ChromaDB not found
- **Solution**: Check that RAG systems have been initialized with data

**Issue**: Module import errors
- **Solution**: Run from project root directory or check sys.path configuration

## Notes

- The evaluation system uses **Gemini 2.0 Flash Experimental** as the judge for better reasoning capabilities
- Results are saved incrementally to prevent data loss
- The system respects API rate limits with configurable delays
- Ground truth answers are based on actual ingested data in ChromaDB and Neo4j
