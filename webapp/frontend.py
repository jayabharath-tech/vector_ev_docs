"""
Streamlit frontend for RAG chatbot.
User-friendly interface for uploading PDFs and chatting with the RAG agent.
"""
import os
import json
import requests
import streamlit as st
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HISTORY_FILE = Path(".streamlit_chat_history.json")  # Persist conversation locally

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


def chat_with_backend(question, conversation_history):
    """Send a chat request to the FastAPI backend."""
    try:
        payload = {
            "question": question,
            "conversation_history": conversation_history
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


def check_backend_health():
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def load_conversation_history():
    """Load conversation history from file if it exists."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_conversation_history(history):
    """Save conversation history to file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.warning(f"Could not save conversation history: {e}")


# ============================================================================
# SESSION STATE
# ============================================================================

if "conversation_history" not in st.session_state:
    # Start with fresh history for each session
    st.session_state.conversation_history = []

if "source_metadata_cache" not in st.session_state:
    st.session_state.source_metadata_cache = {}  # Store sources by message index

if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = check_backend_health()

if "uploaded_files_cleared" not in st.session_state:
    st.session_state.uploaded_files_cleared = False

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
    st.subheader("💬 Conversation")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.conversation_history = []
        save_conversation_history([])
        st.success("Chat history cleared")
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

st.title("🤖 Chatbot")
st.markdown("Ask questions about your uploaded documents")

# Display chat history
chat_container = st.container()
with chat_container:
    for idx, message in enumerate(st.session_state.conversation_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources for assistant messages with cached metadata
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
col1, col2 = st.columns([1, 0.15])

with col1:
    user_input = st.chat_input(
        "Ask a question...",
        disabled=not st.session_state.backend_healthy
    )

if user_input:
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_input
    })
    save_conversation_history(st.session_state.conversation_history)

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from backend
    with st.chat_message("assistant"):
        if not st.session_state.backend_healthy:
            st.error(f"❌ Backend is not responding at {BACKEND_URL}")
            response = None
        else:
            with st.spinner("🔍 Searching knowledge base..."):
                # Convert conversation history to backend format
                conversation_for_api = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.conversation_history[:-1]  # Exclude current user message
                ]

                response = chat_with_backend(user_input, conversation_for_api)

            # Check if we got a valid response
            if response:
                # Display answer
                st.markdown(response["answer"])

                # Add assistant response to history
                message_index = len(st.session_state.conversation_history)
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
                save_conversation_history(st.session_state.conversation_history)

                # Store source metadata for this assistant message
                if response.get("source_metadata"):
                    st.session_state.source_metadata_cache[message_index] = {
                        "source_metadata": response["source_metadata"],
                        "source_snippets": response["source_snippets"]
                    }

                # Rerun to show updated chat history
                st.rerun()
            else:
                st.error("Failed to get response from backend")

