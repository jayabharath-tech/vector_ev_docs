"""
FastAPI backend for RAG chatbot.
Simple wrapper around existing RAG logic.

Run from vector_ev_docs directory:
  python -m webapp.backend

Or from base directory:
  python -m rag.vector_ev_docs.webapp.backend
"""
import os
import sys
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# IMPORT RAG SYSTEM
# ============================================================================
# Add parent directory to path for imports
# This allows the script to work from both:
# - vector_ev_docs/ directory: python -m webapp.backend
# - base directory: python -m rag.vector_ev_docs.webapp.backend
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from main import ingest_pdf, main as rag_main

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[ConversationMessage]] = None


class SourceMetadataResponse(BaseModel):
    file_name: str
    page_number: int
    chunk_index: int
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    source_snippets: List[str]
    source_metadata: List[SourceMetadataResponse]


class UploadResponse(BaseModel):
    status: str
    message: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    message: str


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieve-Augmented Generation chatbot with PDF upload",
    version="1.0.0"
)

# Add CORS middleware to allow Streamlit frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running"
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector database.

    The PDF is temporarily saved, ingested, and then deleted.
    Vector embeddings are stored in the shared vector DB for all users.

    Args:
        file: PDF file to upload

    Returns:
        Upload status with number of chunks created
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        # Create temp directory if it doesn't exist
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded file temporarily
        temp_path = temp_dir / file.filename
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)

        # Ingest PDF to vector DB
        await ingest_pdf(str(temp_path))

        # Clean up temp file
        temp_path.unlink()

        # Return success (chunk count is approximate)
        file_size_kb = len(content) / 1024
        estimated_chunks = max(int(file_size_kb / 10), 1)  # Rough estimate

        return UploadResponse(
            status="success",
            message=f"File '{file.filename}' successfully ingested",
            chunks=estimated_chunks
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting file: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for multi-turn conversations.

    Accepts a user question and optional conversation history.
    Returns an answer grounded in the vector DB with source attribution.

    Args:
        request: ChatRequest with question and optional conversation history

    Returns:
        ChatResponse with answer, source snippets, and metadata
    """
    try:
        question = request.question

        # TODO: In a more sophisticated system, pass conversation history to the agent
        # For now, each question is treated independently
        # conversation_history is available in request.conversation_history for future enhancement

        # Run RAG agent
        response = await rag_main(question)

        # Convert SourceMetadata objects to response format
        source_metadata_list = [
            SourceMetadataResponse(
                file_name=meta.file_name,
                page_number=meta.page_number,
                chunk_index=meta.chunk_index,
                relevance_score=meta.relevance_score
            )
            for meta in response.source_metadata
        ]

        return ChatResponse(
            answer=response.answer,
            source_snippets=response.source_snippet,
            source_metadata=source_metadata_list
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", 8000))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")

    print(f"🚀 Starting RAG Chatbot API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
