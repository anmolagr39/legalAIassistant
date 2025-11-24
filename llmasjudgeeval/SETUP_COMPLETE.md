# 🎯 LLM-as-a-Judge Evaluation System - Complete Setup

## 📦 What You Have

Your `llmasjudgeeval` folder now contains a complete evaluation system:

### Core Files Created:

1. **`ground_truth_dataset.json`** ✅
   - 30 questions with ground truth answers
   - Based on actual data from your 3 RAGs and KG
   - Categories: IPC_Section, Constitutional_Law, Past Cases, etc.

2. **`llm_judge_evaluator.py`** ✅
   - Main evaluation engine
   - LLM judge using Gemini 2.0 Flash Exp
   - Evaluates on 4 metrics: Correctness, Completeness, Relevance, Legal Reasoning

3. **`run_evaluation.py`** ✅
   - Quick test scripts
   - Category-specific evaluation
   - List categories

4. **`analyze_results.py`** ✅
   - Detailed analysis of results
   - Category breakdown
   - Common weaknesses identification

5. **`visualize_results.py`** ✅
   - Create charts and graphs
   - Score distributions, heatmaps
   - Category performance plots

6. **`test_setup.py`** ✅
   - Verify system setup
   - Check environment, files, packages
   - Test connections (Neo4j, Gemini API)

7. **`master_eval.py`** ✅
   - Interactive menu interface
   - All-in-one control script
   - Easy command execution

8. **`read_ingested_data.py`** ✅
   - Read data from ChromaDB and Neo4j
   - Used to create ground truth dataset

### Documentation:

9. **`README.md`** ✅ - Full documentation
10. **`QUICKSTART.md`** ✅ - 5-minute quick start guide
11. **`requirements_eval.txt`** ✅ - Python package requirements

---

## 🚀 How to Use

### Method 1: Interactive Mode (Easiest)

```bash
cd d:\legalkg\llmasjudgeeval
python master_eval.py
```

Then use menu commands:
- `test` - Check setup
- `quick 5` - Test with 5 questions
- `full` - Full evaluation
- `analyze` - Analyze results
- `help` - Show all commands

### Method 2: Direct Commands

```bash
# Test setup
python test_setup.py

# Quick evaluation (3 questions)
python run_evaluation.py --quick 3

# Full evaluation (30 questions)
python llm_judge_evaluator.py

# Analyze results
python analyze_results.py evaluation_results/evaluation_report.json

# Create visualizations
python visualize_results.py evaluation_results/evaluation_report.json
```

### Method 3: Category-Specific

```bash
# List categories
python run_evaluation.py --list-categories

# Test one category
python run_evaluation.py --category IPC_Section
```

---

## 📊 Evaluation Metrics

Each response is scored 0-10 on:

1. **Correctness** - Legal facts, case names, sections accurate?
2. **Completeness** - All key points covered?
3. **Relevance** - Answers the question directly?
4. **Legal Reasoning** - Sound reasoning and citations?

**Pass Threshold**: Overall score ≥ 6.0/10

---

## 📁 What Gets Generated

After running evaluation:

```
llmasjudgeeval/
├── evaluation_results/
│   ├── intermediate_results_TIMESTAMP.json  # Progress saved
│   ├── evaluation_report.json               # Main results
│   ├── summary_report_TIMESTAMP.txt         # Human-readable
│   ├── detailed_analysis_report.txt         # After analysis
│   └── visualizations/                      # After visualization
│       ├── score_distribution.png
│       ├── category_performance.png
│       ├── metric_comparison.png
│       └── ...
```

---

## ⚙️ How It Works

```
1. Load Question from ground_truth_dataset.json
         ↓
2. Query agentic_orchestrator_v2.py
         ↓
3. Get Response from orchestrator
         ↓
4. Send to LLM Judge (Gemini 2.0 Flash Exp)
   - Question
   - Ground Truth Answer
   - Generated Answer
         ↓
5. LLM Judge Evaluates:
   - Correctness (0-10)
   - Completeness (0-10)
   - Relevance (0-10)
   - Legal Reasoning (0-10)
   - Overall Score (average)
   - Verdict (PASS/FAIL)
         ↓
6. Save Results (JSON + Text Reports)
         ↓
7. Generate Statistics & Analysis
```

---

## 🎯 Quick Start Commands

```bash
# 1. Check everything is ready
python test_setup.py

# 2. Run a quick 3-question test
python run_evaluation.py --quick 3

# 3. Check results
cd evaluation_results
cat summary_report_*.txt

# 4. If looks good, run full evaluation
cd ..
python llm_judge_evaluator.py
```

---

## 💡 Tips

1. **Start Small**: Always test with `--quick 3` first
2. **Watch Progress**: Results saved after each question (safe to interrupt)
3. **Adjust Delays**: If rate limits, increase `delay_seconds` in scripts
4. **Review Failures**: Check detailed_analysis_report.txt for insights
5. **Iterate**: Improve orchestrator, re-run, compare scores

---

## 📈 Expected Results

For a well-functioning system:
- **Pass Rate**: 70-90%
- **Average Score**: 7-8/10
- **Correctness**: 7.5-8.5/10
- **Completeness**: 6.5-7.5/10
- **Relevance**: 8-9/10
- **Legal Reasoning**: 7-8/10

Lower scores indicate areas for improvement in:
- RAG retrieval quality
- KG query accuracy
- Answer synthesis
- Citation handling

---

## 🔧 Troubleshooting

### Setup Issues

**"GEMINI_API_KEY not found"**
- Add to `d:\legalkg\.env`: `GEMINI_API_KEY=your_key`

**"Neo4j connection failed"**
- Start Neo4j: `neo4j console` or check Windows service
- Verify credentials in `.env`

**"ChromaDB not found"**
- Ensure RAG systems initialized
- Check `3rags/*/chroma_db/` folders exist

### Runtime Issues

**"Rate limit exceeded"**
- Increase delay: Change `delay_seconds=3.0` in scripts
- Use Gemini Flash Lite: `judge_model="gemini-2.0-flash-lite"`

**"Module not found"**
- Install: `pip install -r requirements_eval.txt`
- Run from project root: `cd d:\legalkg`

**"Orchestrator import failed"**
- Check `agentic_orchestrator_v2.py` exists in parent folder
- Verify no syntax errors in orchestrator

---

## 📚 Files You Can Modify

### Customize Evaluation:

**`llm_judge_evaluator.py`:**
```python
# Line ~30: Change judge model
judge_model = "gemini-2.0-flash-exp"  # or "gemini-1.5-pro"

# Line ~280: Adjust delay
delay_seconds = 2.0  # Increase if rate limits

# Line ~50: Modify evaluation prompt
evaluation_prompt = f"""..."""  # Customize judging criteria
```

**`ground_truth_dataset.json`:**
- Add your own questions
- Modify categories
- Update ground truth answers

---

## 🎓 Understanding Results

### Sample Output:
```
[1/30] Evaluating Question 1
  → Querying orchestrator... (3.2s)
  → Systems used: knowledge_graph, ipc_rag
  → Evaluating with LLM judge... (2.1s)
  → Verdict: PASS (Score: 8.5/10)
```

### Result Files:

**`evaluation_report.json`:**
```json
{
  "metadata": {...},
  "statistics": {
    "pass_rate": 80.0,
    "average_overall_score": 7.5,
    "category_breakdown": {...}
  },
  "results": [...]
}
```

**`summary_report.txt`:**
```
OVERALL RESULTS
Total Questions: 30
Passed: 24 (80.0%)
Failed: 6 (20.0%)

AVERAGE SCORES
Correctness:     8.2/10
Completeness:    7.1/10
...
```

---

## 🚀 Next Steps

1. **Run Test**: `python test_setup.py`
2. **Quick Eval**: `python run_evaluation.py --quick 5`
3. **Analyze**: Check `evaluation_results/`
4. **Improve**: Based on failed questions
5. **Re-evaluate**: Track improvement over time
6. **Full Run**: `python llm_judge_evaluator.py`

---

## ✅ System Ready!

Your LLM-as-a-judge evaluation system is now complete and ready to use.

**Start with**: `python master_eval.py` for interactive mode

**Or**: `python test_setup.py` to verify setup

Good luck with your evaluation! 🎉
