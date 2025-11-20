# Legal Knowledge Graph

A high-quality knowledge graph built from Indian legal documents using LLMs (Gemini 2.5 Flash) and Neo4j.

## 🎯 Overview

This project constructs a comprehensive legal knowledge graph from:
- **FIR Dataset**: 3,481 IPC sections with offenses and punishments
- **Case Documents**: 2,914+ Supreme Court judgments
- **Constitution of India**: Constitutional articles and provisions
- **Legal Acts**: Various Indian legal statutes

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Neo4j Desktop** (installed and running)
3. **Gemini API Key** ([Get it here](https://makersuite.google.com/app/apikey))

### Installation

```bash
# 1. Clone/navigate to project directory
cd d:\legalkg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Run setup validation
python setup.py

# 5. Build knowledge graph (300 cases for initial test)
python main.py --max-cases 300
```

## 📁 Project Structure

```
legalkg/
├── config/              # Configuration and Neo4j connection
├── data_processing/     # Data loading and preprocessing
├── extraction/          # LLM-based entity extraction
├── kg_construction/     # Knowledge graph builder
├── main.py             # Main pipeline orchestrator
├── setup.py            # Setup validation script
└── requirements.txt    # Python dependencies
```

## 🔧 Configuration

Edit `.env` file:

```env
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Processing
BATCH_SIZE=10
MAX_CASES=300
```

## 📊 Knowledge Graph Schema

### Node Types
- **IPCSection**: IPC sections with offenses/punishments
- **Case**: Court cases and judgments
- **Court**: Courts (Supreme Court, High Courts)
- **Judge**: Judges who delivered judgments
- **Article**: Constitutional articles
- **LegalAct**: Legal statutes and acts
- **Party**: Plaintiffs, defendants
- **Offense**: Types of offenses
- **Punishment**: Types of punishments
- **LegalPrinciple**: Legal doctrines and precedents

### Relationship Types
- **CITES**: Case citations
- **GOVERNED_BY**: Case governed by IPC/Act
- **APPLIES_TO**: Section applies to offense
- **PRESCRIBES**: Section prescribes punishment
- **HEARD_IN**: Case heard in court
- **DECIDED_BY**: Case decided by judge
- **REFERS_TO**: References to articles
- **INTERPRETS**: Case interprets article/section

## 🎮 Usage

### Basic Usage

```bash
# Process 300 cases (recommended for initial test)
python main.py --max-cases 300

# Process all IPC sections only
python main.py --max-cases 0 --skip-constitution

# Full processing with all data
python main.py --max-cases 2914
```

### Advanced Options

```bash
# Clear database before starting
python main.py --clear-db --max-cases 300

# Skip constitution processing
python main.py --skip-constitution --max-cases 300

# Get help
python main.py --help
```

## 📈 Testing Components

```bash
# Test configuration
python config/settings.py

# Test Neo4j connection
python config/neo4j_config.py

# Test data preprocessing
python data_processing/preprocessor.py

# Test PDF processing
python data_processing/pdf_processor.py

# Test Gemini extractor
python extraction/gemini_extractor.py

# Test KG builder
python kg_construction/kg_builder.py
```

## 🔍 Query Examples

Access Neo4j Browser at `http://localhost:7474` and run:

```cypher
// View all node types and counts
MATCH (n)
RETURN labels(n)[0] as NodeType, count(*) as Count
ORDER BY Count DESC

// Find cases involving IPC Section 302
MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection {section_number: "302"})
RETURN c.case_name, c.date

// Find citation network
MATCH (c1:Case)-[:CITES]->(c2:Case)
RETURN c1.case_name, c2.case_name
LIMIT 20

// Find cases by court
MATCH (c:Case)-[:HEARD_IN]->(court:Court)
WHERE court.name = "Supreme Court of India"
RETURN c.case_name, c.date
LIMIT 10

// Find cases referencing Article 14
MATCH (c:Case)-[:REFERS_TO]->(a:Article {article_number: "14"})
RETURN c.case_name, c.date
```

## 💰 Cost Estimate

For initial test (300 cases):
- **IPC Sections**: ~$2-3
- **Cases (300)**: ~$15-20
- **Constitution**: ~$5-10
- **Total**: ~$25-35

For full processing (2,914 cases):
- **Total**: ~$100-150

## 🐛 Troubleshooting

### Neo4j Connection Failed
```bash
# Check if Neo4j is running
# Verify NEO4J_PASSWORD in .env
# Ensure NEO4J_URI is correct
```

### Gemini API Errors
```bash
# Check API key is set correctly
# Verify you have API quota available
# Check internet connection
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📝 Logging

Logs are written to:
- **Console**: Real-time progress
- **File**: `kg_construction.log`

## 🎯 Next Steps (GraphRAG)

After building the KG:
1. Add vector embeddings to nodes
2. Implement hybrid retrieval (graph + vector search)
3. Build RAG query engine
4. Create chatbot interface

## 📚 Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Knowledge Graphs Guide](https://neo4j.com/developer/knowledge-graph/)

## ⚖️ License

This project processes publicly available Indian legal documents.

## 🤝 Contributing

Contributions welcome! Please test changes before submitting.

---

Built with ❤️ for the legal community
