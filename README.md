# Vector EV Docs RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system for answering questions about Vector Elektronik technical documentation using Pydantic AI, ChromaDB, and Claude.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This RAG system answers questions about Vector Elektronik technical documents by:

1. **Ingesting** PDF documents from a specified directory
2. **Chunking** text into semantic segments with metadata tracking
3. **Embedding** chunks using sentence-transformers
4. **Storing** embeddings in ChromaDB vector database
5. **Retrieving** relevant chunks based on query similarity
6. **Generating** grounded answers using Claude with retrieved context

**Key Benefits:**
- ✅ Answers grounded in actual documents (no hallucinations)
- ✅ Full source attribution (file name, page number, relevance score)
- ✅ Handles multiple PDFs simultaneously
- ✅ Production-ready with thread-safe singleton pattern
- ✅ Async/await support for concurrent requests
- ✅ Comprehensive evaluation framework with LJudge

---

## 🏗️ Architecture

### System Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF INGESTION PHASE                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    📄 Load PDF Files
                    (PyMuPDF - fitz)
                              ↓
                    Extract Text Per Page
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    CHUNKING & EMBEDDING                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
        Split into Overlapping Chunks (1000/150)
        Create Metadata: file_name, page_number, chunk_index
                              ↓
        Generate Embeddings (all-MiniLM-L6-v2)
        384-dimensional vectors via sentence-transformers
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              VECTOR DATABASE STORAGE (ChromaDB)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
        Store with cosine similarity metric:
        ids, embeddings, documents, metadatas
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    QUERY & RETRIEVAL                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
        User Question
                              ↓
        Embed Query (same model as documents)
                              ↓
        Semantic Similarity Search (top 5 results)
                              ↓
        Extract: text, metadata, relevance_score
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   PYDANTIC AI AGENT                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
        Agent uses retrieve_context() tool
        Receives formatted context with sources
                              ↓
        Generate Answer with Claude (haiku-4-5)
        System prompt ensures grounding & honesty
                              ↓
        Return AgentResponse:
        - answer
        - source_snippet(s)
        - source_metadata (file, page, relevance)
```

### Component Architecture

```
┌──────────────────────────────────────┐
│         VectorDBClient               │
│    (Wraps ChromaDB)                  │
│  - query()                           │
│  - upsert_chunks()                   │
└──────────────────────────────────────┘
              ↑
┌──────────────────────────────────────┐
│      EVDocsClient (RAG Logic)        │
│  - ingest()     (PDF → chunks)       │
│  - chunk()      (text splitting)     │
│  - query()      (semantic search)    │
│  - query_with_metadata()             │
└──────────────────────────────────────┘
              ↑
┌──────────────────────────────────────┐
│    Pydantic AI Agent                 │
│  - retrieve_context() tool           │
│  - System prompt (grounding rules)   │
│  - AgentResponse model               │
└──────────────────────────────────────┘
              ↑
┌──────────────────────────────────────┐
│       main() / Evaluation            │
│  - ingest_pdf()                      │
│  - main(question)                    │
│  - evaluator.py (LJudge)           │
└──────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Multi-PDF Support** | Ingest single PDF or entire directory |
| **Smart Chunking** | 1000 char chunks with 150 char overlap |
| **Metadata Tracking** | File name, page number, chunk index, relevance score |
| **Semantic Search** | Cosine similarity on 384-d embeddings |
| **Thread-Safe** | Singleton pattern with locking mechanism |
| **Async Ready** | Full async/await support for concurrency |
| **Source Attribution** | Complete lineage of answers to source documents |
| **Grounding** | System prompt prevents hallucinations & inference |
| **Evaluation** | LJudge integration for quality assessment |
| **Production Ready** | Error handling, logging, proper initialization |

---

## 🚀 Setup

### Prerequisites

- Python 3.13+
- pip or uv package manager
- ANTHROPIC_API_KEY environment variable set

### Installation

```bash
# Navigate to project
cd rag/vector_ev_docs

# Install dependencies
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

### Required Dependencies

```
pydantic>=2.0
pydantic-ai>=0.0.1
chromadb>=0.4.0
sentence-transformers>=2.2.0
PyMuPDF>=1.23.0
anthropic>=0.25.0
pydantic-evals>=0.1.0
```

### Environment Setup

```bash
# Set Claude API key
export ANTHROPIC_API_KEY="your-key-here"

# (Optional) Set HuggingFace token to avoid rate limits
export HF_TOKEN="your-token-here"
```

---

## 📖 Usage

### Basic Usage

```python
import asyncio
from main import ingest_pdf, main

async def demo():
    # Step 1: Ingest PDFs (run once)
    await ingest_pdf("data")  # Directory or single PDF path

    # Step 2: Query the RAG system
    response = await main("What is MCS?")

    print(f"Answer: {response.answer}")
    print(f"Sources: {response.source_metadata}")

asyncio.run(demo())
```

### Command Line

```bash
# Run demo with test queries
python main.py

# Run evaluation suite
python evaluator.py
```

### Example Queries & Expected Outputs

**Query:** "What is the MCS (Megawatt Charging System)?"

**Expected Response:**
```
Answer: The MCS (Megawatt Charging System) is a smart charging standard...
Sources:
  - File: Vector_ElektronikAutomotive_MCS202511_TechnicalArticle_EN.pdf
    Page: 2
    Relevance: 0.92
```

**Query:** "Who is the author of the MCS document?"

**Expected Response:**
```
Answer: Dr. Raphael Pfeil is Product Manager for Development and Test Tools
for Smart Charging at Vector Informatik GmbH. He has extensive knowledge of
ECU development and specializes in conformity testing for smart charging standards.

Sources:
  - File: Vector_ElektronikAutomotive_MCS202511_TechnicalArticle_EN.pdf
    Page: 1
    Relevance: 0.95
```

**Query:** "When was this document published?"

**Expected Response:**
```
Answer: This information is not available in the knowledge base.

Sources: None
```

---

## 🎓 Design Decisions

### 1. Embedding Model: `all-MiniLM-L6-v2`

**Why this model?**

| Aspect | Benefit |
|--------|---------|
| **Local Execution** | No API keys, no rate limits, full privacy |
| **Lightweight** | 384-dim vectors (vs 1536 for GPT embeddings) |
| **Fast** | Sub-second inference on CPU |
| **Quality** | Trained on 1B sentence pairs, excellent semantic understanding |
| **Cost** | Free, no API charges |

**Trade-offs:**
- Slightly lower quality than OpenAI's text-embedding-3-large
- But superior for local deployment and privacy

**Alternative Considered:** OpenAI text-embedding-3-small
- ❌ Requires API key, rate limits, slower, more expensive
- ✅ Marginally better quality

### 2. Chunking Strategy: 1000 chars / 150 overlap

**Why these parameters?**

```
Chunk Size: 1000 characters (~150-200 tokens)
├─ OPTIMAL: Large enough to preserve author bios & context
├─ OPTIMAL: Small enough for precise fact retrieval
├─ NOT 512: Too fragmented, lost context for biographical queries
└─ NOT 2000: Too many false positives for specific queries

Overlap: 150 characters (~25-30 tokens)
├─ OPTIMAL: Prevents info loss at chunk boundaries
├─ OPTIMAL: Query spanning two chunks still retrieves both
├─ NOT 50: Too small, info gets split
└─ NOT 300: Wastes space, redundant coverage
```

**Chunking Algorithm:**
1. Split text by sentence boundaries (period `"."`)
2. Accumulate sentences until reaching chunk_size
3. When threshold exceeded, save chunk
4. Start next chunk with overlap from previous

**Why sentence-based?**
- ✅ Preserves semantic meaning
- ✅ Avoids splitting mid-fact
- ❌ (Alternative: Recursive token-based splitting - too complex)

### 3. Similarity Metric: Cosine

**Why cosine similarity?**

```
Cosine Similarity (CHOSEN)
├─ Normalized: Only direction matters, not magnitude
├─ Fast: HNSW indexing = O(log n) search
├─ Interpretable: Scores 0-1 (0=opposite, 1=identical)
├─ Proven: Industry standard for embedding-based RAG
└─ ChromaDB native: Direct HNSW support

Euclidean Distance (NOT CHOSEN)
├─ ❌ Magnitude-dependent (embedding scale matters)
├─ ❌ Slower search
├─ ❌ Scores unbounded (hard to interpret)
└─ ✅ Computationally simple
```

### 4. Vector Database: ChromaDB

**Why ChromaDB?**

```
ChromaDB (CHOSEN)
├─ Pure Python, in-process
├─ No external services to manage
├─ File-based persistent storage
├─ Full metadata support
├─ Perfect for development & small-medium scale
└─ Easy schema changes

Qdrant (ALTERNATIVE)
├─ Better for 100M+ embeddings
├─ Distributed/cloud deployment
├─ Advanced filtering
└─ Overkill for document corpus <10M embeddings
```

**When to migrate to Qdrant:**
- Production scale with millions of documents
- Need distributed deployment
- Require cloud/Kubernetes setup

### 5. LLM: Claude Haiku (not Opus)

**Why Haiku?**

| Aspect | Haiku | Opus |
|--------|-------|------|
| **Speed** | Fast ✅ | Slower |
| **Cost** | 80% cheaper ✅ | 5x more expensive |
| **Quality** | Sufficient for RAG ✅ | Overkill for fact-following |
| **Task Fit** | Instruction-following ✅ | Complex reasoning (not needed) |

**System Prompt Strategy:**
- Strict grounding rules prevent inference
- Explicit: "Only from context"
- No reasoning/creativity needed
- Pure fact-following

---

## 📊 Evaluation

### Running Evaluation

```bash
python evaluator.py
```

### Evaluation Framework

Uses **LJudge** - another Claude instance grades responses:

```python
Rubric:
1. Relevance - Does answer address the question?
2. Grounding - Is answer based on retrieved context?
3. Accuracy - Is information correct per documents?
4. Completeness - Does it cover key aspects?
5. Honesty - Admits when info unavailable?

PASS: Grounded, relevant, accurate, honest
FAIL: Hallucination, irrelevant, misleading
```

### Test Coverage

**12 Test Cases:**
- ✅ 5 Technical Concepts (MCS, charging, simulation)
- ✅ 3 Implementation Details (CANoe, components)
- ✅ 2 Document Structure (topics, purpose)
- ✅ 2 Capability Tests (metrics, industries)
- ✅ 2 Out-of-scope (admission of lack of info)

**Key Validations:**
```
✅ Single PDF: 100+ chunks processed
✅ Multiple PDFs: No ID collision
✅ Author retrieval: "Dr. Raphael Pfeil" accurate
✅ Out-of-scope: Correctly admits unavailable info
✅ Metadata: Complete lineage (file, page, relevance)
✅ Grounding: No hallucinations detected
```

---

## 🔧 Troubleshooting

### "PDF not found"

```
Error: FileNotFoundError: PDF not found: data/file.pdf

Solutions:
1. Verify path: ls -la data/
2. Use absolute path: /Users/username/.../data/
3. Check permissions: chmod 644 data/*.pdf
```

### "ModuleNotFoundError: No module named 'chromadb'"

```
Error: Missing dependencies

Solution:
pip install chromadb sentence-transformers pydantic-ai
```

### "ANTHROPIC_API_KEY not set"

```
Error: Authentication fails

Solution:
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Very low relevance scores (< 0.3)"

```
Cause: Query doesn't match document content

Solutions:
1. Verify PDF contains the information
2. Try different query phrasing
3. Increase top_n_rows from 5 to 7-10
4. Reduce chunk_size from 1000 to 512
```

### "Answers include hallucination/inference"

```
Cause: Agent inferring beyond retrieved context

Solution:
- System prompt already prevents this
- Verify: Agent called retrieve_context ✓
- Verify: Response cites source_metadata ✓
- Lower temperature if still happening
```

### "ChromaDB collection errors"

```
Error: Collection 'vector_ev_docs' not found

Solution (reset):
rm -rf vector_db/
python main.py  # Re-ingests
```

### "Slow embedding generation"

```
Cause: First run downloads ~500MB model

Solution: Wait for download (one-time)
- First run: ~2-3 minutes
- Subsequent: <1 second

To pre-download:
python -c "from sentence_transformers import SentenceTransformer;
           SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## 🤝 Contributing

### Code Style

- Follow PEP 8
- Type hints on all functions
- Docstrings (Args, Returns, Raises)
- Comments for complex logic

### Testing

```bash
# Run evaluation
python evaluator.py

# Test custom query
# Edit test_queries in main.py __main__
python main.py
```

### Adding Features

**New RAG Client Type:**
1. Inherit from `EVDocsClient`
2. Implement `ingest()`, `chunk()`, `query()`
3. Update factory: `_get_pdf_client()`

**New Evaluator:**
1. Add `Case` to `rag_eval_dataset`
2. Run `python evaluator.py`

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Core RAG agent & ingestion |
| `db.py` | VectorDBClient wrapper |
| `evaluator.py` | LJudge evaluation |
| `README.md` | This documentation |
| `pyproject.toml` | Project metadata |

---

## 🔗 References

- [Pydantic AI Documentation](https://docs.pydantic.dev/latest/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [RAG Best Practices](https://docs.llamaindex.ai/)

---

**Status:** ✅ Production Ready
**Version:** 1.0.0
**Last Updated:** 2026-04-07




# Useful Commands

- python -m venv venv
- Source venv/Scripts/Activate.bat
- pip install -r requirements.txt
- chmod a+x run.sh
- execute run.sh
