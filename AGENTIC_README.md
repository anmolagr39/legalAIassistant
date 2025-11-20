# Agentic Legal Assistant

Multi-system legal research assistant with intelligent query routing.

## 🎯 Architecture

```
User Query
    ↓
[Query Router (LLM)]  ← Analyzes query type & selects systems
    ↓
┌───────────────────────────────────────────────────┐
│  Parallel System Queries                          │
├───────────────────────────────────────────────────┤
│ 1. Knowledge Graph (Neo4j)                        │
│    - 1482 cases, 445 IPC sections, 16 articles   │
│    - Structured queries, relationships, citations │
│                                                    │
│ 2. IPC RAG (ChromaDB)                            │
│    - 445 IPC sections with detailed text         │
│    - Semantic search for offense definitions      │
│                                                    │
│ 3. Past Cases RAG (ChromaDB)                     │
│    - 2914 Supreme Court judgments                 │
│    - Full case text, precedents, reasoning        │
│                                                    │
│ 4. Legal Acts RAG (ChromaDB)                     │
│    - Constitution of India                        │
│    - Legal act provisions                         │
└───────────────────────────────────────────────────┘
    ↓
[Answer Synthesizer (LLM)]  ← Combines results
    ↓
Comprehensive Answer to User
```

## 🚀 Features

### Intelligent Routing
- **LLM-based query analysis** determines which systems to use
- **Query type classification**: factual, analytical, research, comparison
- **Multi-system queries** for comprehensive answers

### Available Systems

1. **Knowledge Graph** (Neo4j)
   - Fast structured queries
   - Citation networks
   - Judge-case relationships
   - IPC section lookups

2. **IPC RAG**
   - Detailed IPC explanations
   - Offense types and punishments
   - Bailable/cognizable status

3. **Past Cases RAG**
   - Case law research
   - Legal precedents
   - Judicial reasoning

4. **Legal Acts RAG**
   - Constitutional provisions
   - Fundamental rights
   - Legal act details

## 📦 Installation

```bash
# Ensure you're in the legalkg directory
cd D:\legalkg

# Install dependencies (if not already installed)
pip install google-generativeai chromadb neo4j python-dotenv rich

# Make sure .env has your API key
# GEMINI_API_KEY=your_key_here
```

## 🎮 Usage

### Interactive Mode

```bash
python agentic_orchestrator.py
```

Example queries:
- "What is IPC section 302?" → Routes to KG + IPC RAG
- "Cases involving murder" → Routes to KG + Past Cases RAG
- "Explain Article 21" → Routes to KG + Legal Acts RAG
- "Who decided case XYZ?" → Routes to KG only
- "Difference between bailable and non-bailable" → Routes to IPC RAG only

### Sample Session

```
🏛️  AGENTIC LEGAL ASSISTANT
================================================================================

💬 Your question: What is IPC section 302?

🔀 Step 1: Routing query...
  Selected systems: ['knowledge_graph', 'ipc_rag']
  Reasoning: Query asks about specific IPC section - needs both structured data and detailed explanation
  Query type: factual

🔍 Step 2: Querying systems...
  Loading knowledge_graph...
  ✓ knowledge_graph: Retrieved 1 results
  Loading ipc_rag...
  ✓ ipc_rag: Retrieved 5 results

💭 Step 3: Generating answer...

================================================================================
✅ FINAL ANSWER:
================================================================================
IPC Section 302 deals with the punishment for murder...

Sources:
- Knowledge Graph: Section 302 is cited in 45 cases
- IPC RAG: Detailed offense definition and punishment details
================================================================================
```

## 🔧 How It Works

### Phase 1: Query Routing
```python
{
    "systems": ["knowledge_graph", "ipc_rag"],
    "reasoning": "Needs both structured lookup and detailed explanation",
    "query_type": "factual",
    "requires_multiple": true
}
```

### Phase 2: Parallel System Queries
- Each selected system is queried simultaneously
- Results include source attribution
- Errors are handled gracefully

### Phase 3: Answer Synthesis
- LLM combines all retrieved information
- Maintains source attribution
- Resolves conflicts between sources
- Provides comprehensive answer

## 📊 Query Routing Examples

| Query Type | Systems Used | Reasoning |
|-----------|-------------|-----------|
| "What is IPC 302?" | KG + IPC RAG | Quick lookup + detailed explanation |
| "Murder cases" | KG + Past Cases RAG | Structured query + case law |
| "Article 21 cases" | KG + Legal Acts RAG | Graph relationships + full text |
| "Judge XYZ cases" | KG only | Pure graph query |
| "Bailable offenses" | IPC RAG only | Conceptual IPC query |
| "Landmark judgments on privacy" | Past Cases RAG only | Case law research |

## 🎯 System Selection Logic

The router LLM considers:
- **Query intent**: Lookup, research, comparison, analysis
- **Data requirements**: Structured vs. unstructured
- **Information needed**: Quick facts vs. detailed explanation
- **Relationships**: Citations, precedents, connections
- **Specificity**: Exact section/case vs. broad topic

## 🔍 Example Queries

### Factual Queries
```
- "What is IPC section 420?"
- "When was case ABC decided?"
- "What does Article 19 say?"
```

### Research Queries
```
- "Cases on freedom of speech"
- "Precedents for right to privacy"
- "Murder convictions in 2020"
```

### Analytical Queries
```
- "Compare IPC 302 and 304"
- "How have courts interpreted Article 21?"
- "Evolution of Section 377 cases"
```

### Relationship Queries
```
- "Which cases cite Kesavananda Bharati?"
- "Cases decided by Justice XYZ"
- "IPC sections in ABC case"
```

## 📈 Performance

- **Query Routing**: ~2-3 seconds (LLM call)
- **Knowledge Graph**: ~100-500ms (Neo4j)
- **RAG Systems**: ~2-5 seconds each (retrieval + LLM)
- **Total**: ~5-15 seconds for multi-system queries

## 🔄 Future Enhancements

- [ ] Caching for common queries
- [ ] Confidence scores per source
- [ ] Query refinement loop
- [ ] Streaming responses
- [ ] Citation extraction
- [ ] Query history
- [ ] Multi-language support

## 📝 Notes

- All 3 RAG systems must be set up with their vector stores
- Neo4j must be running (bolt://localhost:7687)
- Requires GEMINI_API_KEY in .env
- Systems are loaded lazily (only when needed)

## 🐛 Troubleshooting

**"Failed to initialize RAG engine"**
- Ensure vector stores are created for each RAG system
- Check that data has been ingested

**"Neo4j connection failed"**
- Start Neo4j Desktop
- Verify credentials in .env

**"No results found"**
- Check if data is loaded in the selected systems
- Try rephrasing the query

## 🤝 Contributing

The orchestrator is modular - add new systems by:
1. Adding to `SystemType` enum
2. Creating system interface class
3. Adding to `_get_system()` method
4. Updating router prompt
