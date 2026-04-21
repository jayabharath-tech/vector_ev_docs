"""
FastAPI backend for RAG chatbot with conversation persistence.
Integrates RAG system with SQLite conversation storage + in-memory cache.

Run from vector_ev_docs directory:
  python -m webapp.backend

Or from base directory:
  python -m rag.vector_ev_docs.webapp.backend
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# IMPORT RAG SYSTEM & CONVERSATION MANAGER
# ============================================================================
# Add parent directory to path for imports
# This allows the script to work from both:
# - vector_ev_docs/ directory: python -m webapp.backend
# - base directory: python -m rag.vector_ev_docs.webapp.backend
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from main import ingest_pdf, main as rag_main
from conversation_manager import ConversationManager

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None  # Use existing or generate new
    question: str
    user_id: str = "default_user"  # Default for anonymous users


class SourceMetadataResponse(BaseModel):
    file_name: str
    page_number: int
    chunk_index: int
    relevance_score: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    source_snippets: List[str]
    source_metadata: List[SourceMetadataResponse]


class ConversationData(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ListConversationsResponse(BaseModel):
    conversations: List[ConversationData]
    total: int


class UploadResponse(BaseModel):
    status: str
    message: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    message: str


# ============================================================================
# FASTAPI APP & CONVERSATION MANAGER
# ============================================================================

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieve-Augmented Generation chatbot with PDF upload and conversation persistence",
    version="1.1.0"
)

# Add CORS middleware to allow Streamlit frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize conversation manager (SQLite + in-memory cache)
# Maintains last 3 conversations in memory, archives rest to SQLite
conversation_manager = ConversationManager(
    db_path="./conversations.db",
    max_cache_size=3
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
    Chat endpoint with conversation persistence.

    Creates or loads a conversation, saves user message, gets RAG response,
    and saves assistant message to SQLite + memory cache.

    Args:
        request: ChatRequest with question, conversation_id (optional), user_id

    Returns:
        ChatResponse with conversation_id, answer, sources
    """
    try:
        # Get or create conversation
        if request.conversation_id:
            conversation = conversation_manager.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation not found: {request.conversation_id}"
                )
            conversation_id = request.conversation_id
        else:
            # Generate new conversation ID and create
            conversation_id = str(uuid.uuid4())
            conversation_manager.create_conversation(
                conversation_id=conversation_id,
                user_id=request.user_id,
                title=f"Chat - {request.question[:50]}...",
                metadata={"first_question": request.question}
            )

        # Save user message
        conversation_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.question,
            metadata={"type": "question"}
        )

        # Get conversation context (last 5 messages) for RAG agent
        # Note: Currently the agent doesn't use this, but it's available
        context_messages = conversation_manager.get_context(conversation_id, last_n=5)

        # Run RAG agent with the question
        response = await rag_main(request.question)

        # Save assistant message
        conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.answer,
            metadata={
                "type": "answer",
                "sources": len(response.source_metadata),
                "relevance_scores": [m.relevance_score for m in response.source_metadata]
            }
        )

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
            conversation_id=conversation_id,
            answer=response.answer,
            source_snippets=response.source_snippet,
            source_metadata=source_metadata_list
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


# ============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/conversations")
async def list_conversations(user_id: str = "default_user", limit: int = 10):
    """
    List conversations for a user.

    Args:
        user_id: User ID
        limit: Maximum number of conversations to return

    Returns:
        List of conversations (without messages)
    """
    try:
        conversations = conversation_manager.list_conversations(user_id, limit)
        return ListConversationsResponse(
            conversations=[
                ConversationData(
                    id=conv["id"],
                    user_id=conv["user_id"],
                    title=conv["title"],
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    message_count=len(conv.get("messages", []))
                )
                for conv in conversations
            ],
            total=len(conversations)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing conversations: {str(e)}"
        )


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation with all messages.

    Args:
        conversation_id: Conversation ID

    Returns:
        Conversation data with all messages
    """
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}"
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation: {str(e)}"
        )


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: Conversation ID

    Returns:
        Confirmation message
    """
    try:
        success = conversation_manager.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}"
            )
        return {"status": "deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation: {str(e)}"
        )


@app.get("/status")
async def get_status():
    """Get system status including cache statistics."""
    try:
        cache_stats = conversation_manager.get_cache_stats()
        return {
            "status": "healthy",
            "cache": cache_stats,
            "database": "conversations.db"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status: {str(e)}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", 8000))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")

    print(f"🚀 Starting RAG Chatbot API on {host}:{port}")
    print(f"📁 Conversations stored in: ./conversations.db")
    print(f"💾 In-memory cache size: 3 conversations")


    uvicorn.run(app, host=host, port=port)
