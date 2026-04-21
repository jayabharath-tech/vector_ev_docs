"""
Streamlit frontend for RAG chatbot with session state persistence.
User-friendly interface with conversation history maintained across page refreshes.
Conversations stored in backend SQLite + frontend session state cache.
"""
import os
import json
import requests
import streamlit as st
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_USER_ID = "default_user"  # Could be from auth system

st.set_page_config(
    page_title="EV knowledgeBase",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def upload_file_to_backend(uploaded_file):
    """Upload a file to the FastAPI backend."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return None
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return None


def chat_with_backend(conversation_id, question, user_id=DEFAULT_USER_ID):
    """
    Send a chat request to the backend with conversation persistence.

    Backend automatically:
    - Creates new conversation if not exists
    - Saves user message
    - Generates response
    - Saves assistant message
    - Returns conversation_id with response
    """
    try:
        payload = {
            "conversation_id": conversation_id,
            "question": question,
            "user_id": user_id
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        st.error(f"❌ Backend request timed out. The AI is taking too long to respond.")
        return None
    except Exception as e:
        st.error(f"❌ Chat failed: {str(e)}")
        return None


def load_conversation_from_backend(conversation_id):
    """Load a full conversation from backend (all messages)."""
    try:
        response = requests.get(f"{BACKEND_URL}/conversations/{conversation_id}", timeout=10)
        response.raise_for_status()
        data = response.json()

        # Convert to frontend format
        return {
            "id": data["id"],
            "title": data["title"],
            "messages": data["messages"]
        }
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load conversation: {str(e)}")
        return None


def list_conversations(user_id=DEFAULT_USER_ID, limit=10):
    """List all conversations for a user."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/conversations",
            params={"user_id": user_id, "limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except:
        return {"conversations": [], "total": 0}


def delete_conversation_on_backend(conversation_id):
    """Delete a conversation from backend."""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/conversations/{conversation_id}",
            timeout=10
        )
        response.raise_for_status()
        return True
    except:
        st.error(f"❌ Failed to delete conversation")
        return False


def check_backend_health():
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False




# ============================================================================
# SESSION STATE - Maintains conversation state across page refreshes
# ============================================================================

# Conversation ID (persists across refreshes within session)
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None  # Will be set on first message

# Chat messages (persists across refreshes)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Source metadata cache (persists across refreshes)
if "source_metadata_cache" not in st.session_state:
    st.session_state.source_metadata_cache = {}

# Backend health check
if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = check_backend_health()

# File uploader state
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("📋 EV knowledgeBase")

    st.divider()

    # Upload section
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files to upload",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        help="Upload one or more PDF documents to the shared knowledge base"
    )

    if uploaded_files and st.session_state.backend_healthy:
        if st.button("📁 Upload", use_container_width=True):
            progress_bar = st.progress(0)
            upload_success = True
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    result = upload_file_to_backend(uploaded_file)
                    if result and result.get("status") == "success":
                        st.success(f"✅ {uploaded_file.name} ({result['chunks']} chunks)")
                    else:
                        upload_success = False
                    progress_bar.progress((idx + 1) / len(uploaded_files))

            # Clear the file uploader after successful upload
            if upload_success:
                st.session_state.file_uploader_key += 1
                st.rerun()

    st.divider()

    # Conversation management
    st.subheader("💬 Chat")

    if st.button("✨ New Chat", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.source_metadata_cache = {}
        st.rerun()

    st.divider()

    # Info
    st.subheader("ℹ️ About")
    st.caption(
        "EV knowledgeBase. "
        "Upload PDFs and ask questions about them!"
    )

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Header
col1, col2 = st.columns([1, 0.2])
with col1:
    st.title("🤖 Chatbot")
    st.markdown("Ask questions about your uploaded documents")

# Display chat history
chat_container = st.container()
with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources for assistant messages
            if message["role"] == "assistant" and idx in st.session_state.source_metadata_cache:
                cache_data = st.session_state.source_metadata_cache[idx]
                source_metadata = cache_data["source_metadata"]

                # Extract unique file names
                unique_files = list(dict.fromkeys([meta['file_name'] for meta in source_metadata]))

                st.divider()

                # Show sources as expandable list
                with st.expander("📄 Sources"):
                    for file in unique_files:
                        st.markdown(f"- {file}")

# Chat input
user_input = st.chat_input(
    "Ask a question...",
    disabled=not st.session_state.backend_healthy
)

if user_input:
    # Add user message to session state
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from backend
    with st.chat_message("assistant"):
        if not st.session_state.backend_healthy:
            st.error(f"❌ Backend is not responding at {BACKEND_URL}")
        else:
            with st.spinner("🔍 Searching knowledge base..."):
                response = chat_with_backend(
                    st.session_state.conversation_id,
                    user_input,
                    user_id=DEFAULT_USER_ID
                )

            # Check if we got a valid response
            if response:
                # Display answer
                st.markdown(response["answer"])

                # Update conversation ID if backend returned a new one
                if response.get("conversation_id"):
                    st.session_state.conversation_id = response["conversation_id"]

                # Add assistant response to session state
                message_index = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })

                # Store source metadata for this assistant message
                if response.get("source_metadata"):
                    st.session_state.source_metadata_cache[message_index] = {
                        "source_metadata": response["source_metadata"],
                        "source_snippets": response["source_snippets"]
                    }

                # Rerun to show updated chat
                st.rerun()
            else:
                st.error("Failed to get response from backend")

