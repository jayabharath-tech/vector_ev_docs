#!/bin/bash

# Run both backend and frontend in parallel
# Run this from vector_ev_docs directory
# Usage: cd rag/vector_ev_docs && ./run.sh

echo "🚀 Starting RAG Chatbot..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  ANTHROPIC_API_KEY not set. Please export it:${NC}"
    echo "export ANTHROPIC_API_KEY='your_key_here'"
    exit 1
fi

echo -e "${GREEN}✅ ANTHROPIC_API_KEY is set${NC}"
echo ""

# Verify we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${YELLOW}⚠️  Please run this script from rag/vector_ev_docs directory${NC}"
    echo "cd rag/vector_ev_docs && ./run.sh"
    exit 1
fi

echo "📁 Working directory: $(pwd)"
echo ""

# Load .env if it exists
if [ -f webapp/.env ]; then
    echo "📁 Loading .env configuration..."
    export $(cat webapp/.env | grep -v '#' | xargs)
fi

echo ""
echo "Backend will run on: http://${BACKEND_HOST:-0.0.0.0}:${BACKEND_PORT:-8000}"
echo "Frontend will run on: http://localhost:${STREAMLIT_SERVER_PORT:-8501}"
echo ""

# Start backend in background
echo -e "${GREEN}🔧 Starting Backend (FastAPI)...${NC}"
python -m webapp.backend &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend in background
echo -e "${GREEN}🎨 Starting Frontend (Streamlit)...${NC}"
streamlit run webapp/frontend.py &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}✅ Both services started!${NC}"
echo ""
echo "📍 Frontend: http://localhost:${STREAMLIT_SERVER_PORT:-8501}"
echo "📍 Backend API: http://localhost:${BACKEND_PORT:-8000}/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

trap cleanup SIGINT

# Wait for both processes
wait
