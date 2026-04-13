# RAG Chatbot Web Application

A web-based interface for the Vector EV Docs RAG system with **FastAPI backend** and **Streamlit frontend**.

## Features

- 📄 **PDF Upload** — Upload multiple PDFs to the shared vector database
- 💬 **Multi-turn Chat** — Ask questions with full conversation history
- 📚 **Source Attribution** — View retrieved documents with relevance scores
- 🔄 **Shared Knowledge Base** — All users query the same vector DB
- ⚡ **Real-time Ingestion** — Uploaded PDFs are immediately available for queries

## Architecture

```
Frontend (Streamlit)
    ↓ HTTP Requests
Backend (FastAPI)
    ↓ Async/Await
RAG System (../main.py)
    ↓
Vector DB (../vector_db/)
```

## Quick Start

### Prerequisites

```bash
# From rag/vector_ev_docs/ directory
pip install -r webapp/requirements.txt

# Ensure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY="your_key_here"
```

### Easiest: Run Both Services Together

**From `rag/vector_ev_docs/` directory:**
```bash
chmod +x run.sh
./run.sh
```

This starts both backend and frontend automatically.

---

### Manual: Run Services Separately

**From `rag/vector_ev_docs/` directory - Terminal 1 (Backend):**
```bash
python -m webapp.backend
```

Output:
```
🚀 Starting RAG Chatbot API on 0.0.0.0:8000
Uvicorn running on http://0.0.0.0:8000
```

Visit: http://localhost:8000/docs (Swagger UI)

**From `rag/vector_ev_docs/webapp/` directory - Terminal 2 (Frontend):**
```bash
streamlit run frontend.py
```

Output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://your_ip:8501
```

### Use the Chatbot

1. **Open** http://localhost:8501
2. **Upload PDFs** using the sidebar file uploader
3. **Ask questions** in the chat interface
4. **View sources** from retrieved documents

## Configuration

### Environment Variables

Create `.env` in `webapp/` directory:

```bash
# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend
STREAMLIT_SERVER_PORT=8501

# RAG
RAG_ENABLE_QUERY_EXPANSION=false  # Set true to enable LLM query expansion
VECTOR_DB_PATH=./ev_vector_db

# API
ANTHROPIC_API_KEY=sk-ant-...
```

Or copy from `.env.example`:
```bash
cp .env.example .env
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload PDF
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

Response:
```json
{
  "status": "success",
  "message": "File 'document.pdf' successfully ingested",
  "chunks": 45
}
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is MCS?",
    "conversation_history": []
  }'
```

Response:
```json
{
  "answer": "The MCS (Megawatt Charging System) is...",
  "source_snippets": ["...excerpt..."],
  "source_metadata": [
    {
      "file_name": "Vector_MCS_document.pdf",
      "page_number": 1,
      "chunk_index": 5,
      "relevance_score": 0.92
    }
  ]
}
```

## File Structure

```
webapp/
├── backend.py              # FastAPI server
├── frontend.py             # Streamlit app
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
└── README.md              # This file
```

## How It Works

### Upload Flow
1. User uploads PDF via Streamlit UI
2. Streamlit sends file to `/upload` endpoint
3. FastAPI saves file temporarily
4. Calls `ingest_pdf()` from `../main.py`
5. PDF is chunked, embedded, and stored in shared vector DB
6. Temporary file is deleted

### Chat Flow
1. User asks question in Streamlit
2. Streamlit sends to `/chat` endpoint with conversation history
3. FastAPI calls `rag_main(question)` from `../main.py`
4. RAG agent:
   - Calls `retrieve_context()` (max 3 times)
   - Searches vector DB for relevant chunks
   - Generates answer using Claude
5. Response returned with sources and metadata
6. Streamlit displays answer and sources

## Important Notes

⚠️ **Shared Vector DB** — All users write to the same database
- Concurrent uploads may cause issues (in production, add locking)
- No user isolation
- Perfect for demo/learning

✅ **Iteration Limit** — Agent is capped at 3 retrieve_context calls
- Prevents runaway loops
- Ensures predictable latency

✅ **Query Expansion** — Disabled by default (set `RAG_ENABLE_QUERY_EXPANSION=true` to enable)
- Adds 2-3 seconds latency
- Improves retrieval for ambiguous queries

## Troubleshooting

### ModuleNotFoundError: No module named 'main'
```
ModuleNotFoundError: No module named 'main'
```
**Solution:** Make sure you're running from the **`rag/vector_ev_docs/` directory**:
```bash
cd rag/vector_ev_docs
python -m webapp.backend
```

### Backend won't start
```
Error: [Errno 48] Address already in use
```
Solution: Change `BACKEND_PORT` in `.env` or kill process on port 8000

### Frontend can't connect to backend
```
❌ Cannot connect to backend at http://localhost:8000
```
Solution:
- Make sure backend is running
- Check `BACKEND_URL` in frontend code matches backend host:port
- Verify you're running from the correct directory

### File upload fails
```
❌ Upload failed: Only PDF files are supported
```
Solution: Ensure file is a valid PDF

### Chat returns empty answer
```
No relevant documents found in knowledge base.
```
Solution:
- Upload documents first
- Try different question wording
- Check vector DB exists in `./ev_vector_db/`

## Development Notes

### To disable query expansion for evaluation:
```bash
cd rag/vector_ev_docs
export RAG_ENABLE_QUERY_EXPANSION=false
# Then run the services
./run.sh
```

### To test API directly:
```bash
# Open Swagger UI (after starting backend)
http://localhost:8000/docs
```

### To view uploaded documents:
Vector DB stored in `./ev_vector_db/` (created after first upload)

## Future Enhancements

- [ ] User authentication & isolation
- [ ] Persistent chat history (database)
- [ ] Document management (delete, edit metadata)
- [ ] Advanced filtering (by source, date range)
- [ ] Rate limiting for shared DB
- [ ] Async file uploads for large PDFs
- [ ] Conversation export (PDF, JSON)

## License

Same as parent project
