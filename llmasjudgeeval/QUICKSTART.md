# Quick Start Guide - LLM-as-a-Judge Evaluation

## 🚀 Getting Started in 5 Minutes

### Step 1: Verify Setup
```bash
cd llmasjudgeeval
python test_setup.py
```

This checks:
- ✓ Environment variables (GEMINI_API_KEY, NEO4J_PASSWORD)
- ✓ Required files exist
- ✓ Packages installed
- ✓ Orchestrator import works
- ✓ Neo4j connection
- ✓ Gemini API working

### Step 2: Run Quick Test (3-5 questions)
```bash
python run_evaluation.py --quick 5
```

This will:
- Query your orchestrator with 5 questions
- Evaluate responses with LLM judge
- Show results in ~2-3 minutes

### Step 3: Check Results
```bash
cd evaluation_results
ls
```

You'll see:
- `intermediate_results_TIMESTAMP.json` - Progress saved
- `evaluation_report.json` - Final results with statistics
- `summary_report_TIMESTAMP.txt` - Human-readable summary

### Step 4: Full Evaluation (Optional)
```bash
python llm_judge_evaluator.py
```

Evaluates all 30 questions (~10-15 minutes)

---

## 💡 Alternative: Interactive Mode

For an easy menu-driven interface:

```bash
python master_eval.py
```

Then use commands:
- `test` - Run setup checks
- `quick 5` - Quick 5-question test
- `full` - Full 30-question evaluation
- `list-cat` - See categories
- `category IPC_Section` - Test one category
- `analyze` - Analyze latest results
- `help` - Show all commands

---

## 📊 Example Session

```bash
# 1. Test setup
$ python test_setup.py
✓ ALL CHECKS PASSED

# 2. Quick test
$ python run_evaluation.py --quick 3

[1/3] Evaluating Question 1
  → Querying orchestrator...
  → Got response in 3.2s
  → Verdict: PASS (Score: 8.5/10)

[2/3] Evaluating Question 2
  → Querying orchestrator...
  → Got response in 2.8s
  → Verdict: PASS (Score: 7.2/10)

[3/3] Evaluating Question 3
  → Querying orchestrator...
  → Got response in 3.1s
  → Verdict: FAIL (Score: 5.8/10)

Evaluation Complete!
Passed: 2 (66.7%)
Failed: 1 (33.3%)
Average Score: 7.17/10

# 3. View results
$ cat evaluation_results/summary_report_*.txt

# 4. Analyze
$ python analyze_results.py evaluation_results/evaluation_report.json
```

---

## 🔧 Troubleshooting

### "GEMINI_API_KEY not set"
Add to `.env` file in project root:
```
GEMINI_API_KEY=your_key_here
```

### "Neo4j connection failed"
1. Start Neo4j: `neo4j console`
2. Check credentials in `.env`

### "Rate limit exceeded"
Increase delay in scripts:
```python
delay_seconds=3.0  # Instead of 2.0
```

---

## 📁 What Gets Created

After running evaluation:

```
llmasjudgeeval/
├── evaluation_results/           # Created automatically
│   ├── intermediate_results_*.json
│   ├── evaluation_report.json    # Main results
│   ├── summary_report_*.txt      # Human-readable
│   ├── detailed_analysis_report.txt
│   └── visualizations/           # If you run visualize
│       ├── score_distribution.png
│       ├── category_performance.png
│       └── ...
```

---

## 🎯 Quick Commands Reference

| Command | Description | Time |
|---------|-------------|------|
| `python test_setup.py` | Verify setup | 10s |
| `python run_evaluation.py --quick 3` | Test 3 questions | 1-2 min |
| `python run_evaluation.py --quick 10` | Test 10 questions | 3-5 min |
| `python llm_judge_evaluator.py` | Full 30 questions | 10-15 min |
| `python run_evaluation.py --list-categories` | List categories | instant |
| `python run_evaluation.py --category IPC_Section` | One category | varies |
| `python analyze_results.py <file>` | Analyze results | instant |
| `python visualize_results.py <file>` | Create charts | 10-20s |
| `python master_eval.py` | Interactive mode | - |

---

## 💪 Pro Tips

1. **Start Small**: Always test with `--quick 3` first
2. **Monitor Progress**: Results saved after each question
3. **Check Logs**: Watch terminal for errors
4. **Analyze Fast**: Run analysis while evaluation continues
5. **Compare Runs**: Keep multiple result files to track improvements

---

## 📈 Understanding Scores

- **9-10**: Excellent - All key points, accurate, well-reasoned
- **7-8.9**: Good - Most points covered, minor issues
- **5-6.9**: Fair - Some correct info, significant gaps
- **3-4.9**: Poor - Major inaccuracies or missing info
- **0-2.9**: Very Poor - Mostly incorrect or off-topic

**Pass Threshold**: 6.0/10

---

## 🎓 Next Steps After First Run

1. Review failed questions in detailed analysis
2. Check which categories perform worst
3. Examine common weaknesses
4. Improve orchestrator based on insights
5. Re-run evaluation to measure improvement

---

Need help? Check the full [README.md](README.md) for detailed documentation.
