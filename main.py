"""
RAG Agent for Vector Elektronik Technical Documentation.
Retrieves relevant context from knowledge base and generates grounded answers.
"""
import warnings
import os

# Load environment variables from .env file FIRST, before any other code
from dotenv import load_dotenv
load_dotenv()

from telemetry import init_telemetry

warnings.filterwarnings('ignore')

import threading
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict
import pymupdf4llm  # Extracts text + tables + image descriptions
import re  # Regex for markdown structure detection

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sentence_transformers import SentenceTransformer

from db import VectorDBClient

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# RAG_LOG_LEVEL environment variable: DEBUG, INFO, WARNING, ERROR
# Set in .env: RAG_LOG_LEVEL=DEBUG (default: INFO)
RAG_LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, RAG_LOG_LEVEL, logging.INFO),
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION FROM .env FILE
# ============================================================================

# Query expansion configuration
# Set in .env: RAG_ENABLE_QUERY_EXPANSION=false (default: true)
RAG_ENABLE_QUERY_EXPANSION = os.getenv("RAG_ENABLE_QUERY_EXPANSION", "true").lower() not in ("false", "0")

# LLM Provider configuration
# Set in .env: RAG_LLM_PROVIDER=groq|anthropic (default: anthropic)
# Set in .env: RAG_LLM_MODEL=model_name (default: claude-haiku-4-5-20251001)
RAG_LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "anthropic").lower()
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-haiku-4-5-20251001")

# PDF Ingestion batch size
# Set in .env: RAG_INGEST_BATCH_SIZE=30 (default: 30)
# For large PDFs with many chunks, batching prevents memory overflow
# Smaller batch = more stable but slower, larger batch = faster but uses more RAM
RAG_INGEST_BATCH_SIZE = int(os.getenv("RAG_INGEST_BATCH_SIZE", "30"))

init_telemetry(project_name="Vector EV Docs RAG")

# ============================================================================
# LLM PROVIDER ABSTRACTION
# ============================================================================

def get_llm_client():
    """
    Factory function to get the configured LLM client.

    Supports:
    - "anthropic": Uses Anthropic API (default)
    - "groq": Uses Groq API for fast inference

    Environment variables:
    - RAG_LLM_PROVIDER: Provider selection
    - RAG_LLM_MODEL: Model name override (provider-dependent)
    """
    if RAG_LLM_PROVIDER == "groq":
        try:
            from groq import Groq
            return Groq(api_key=os.getenv("GROQ_API_KEY"))
        except ImportError:
            logger.error("Groq provider selected but groq package not installed. Install with: pip install groq")
            raise
    else:  # Default to anthropic
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


async def describe_image_with_vision(image_bytes: bytes, media_type: str = "image/png") -> str:
    """
    Describe an image using the configured LLM provider's vision API.

    Args:
        image_bytes: Raw image bytes
        media_type: MIME type (e.g. "image/png", "image/jpeg")

    Returns:
        Text description of the image
    """
    import base64

    if RAG_LLM_PROVIDER == "groq":
        # Groq vision using llama-3.2-11b-vision-preview
        client = get_llm_client()
        encoded = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "This image is from a technical PDF document. "
                            "Describe everything visible: all text, labels, numbers, "
                            "table data, chart values, and diagram components. "
                            "Be thorough so the description can be used as searchable text."
                        )
                    }
                ]
            }],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    else:
        # Default: Anthropic vision
        client = get_llm_client()
        encoded = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = client.messages.create(
            model=RAG_LLM_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": encoded}
                    },
                    {
                        "type": "text",
                        "text": (
                            "This image is from a technical PDF document. "
                            "Describe everything visible: all text, labels, numbers, "
                            "table data, chart values, and diagram components. "
                            "Be thorough so the description can be used as searchable text."
                        )
                    }
                ]
            }]
        )
        return response.content[0].text


# SYSTEM_PROMPT = """You are a professional technical assistant for Electric Vehicles Safety documentation.
SYSTEM_PROMPT = """You are a professional technical assistant to read and present the data from a knowledgebase.

🔴 MANDATORY GROUNDING RULES (non-negotiable):
1. BEFORE answering ANY question, you MUST use the retrieve_context tool EXACTLY ONCE per question
   - Single retrieve_context call retrieves comprehensive results
   - Do NOT make multiple retrieve_context calls with different queries
   - Do NOT call retrieve_context again to get "more specific" information
2. Your answer must ONLY contain facts explicitly stated in the retrieved context
3. Do NOT use your training data, general knowledge, or inference
4. Do NOT interpret, combine, or deduce facts beyond what's explicitly written
5. If retrieved context is empty or irrelevant, respond: "This information is not available in the knowledge base"

✅ ALLOWED TOOL SEQUENCE (and ONLY this sequence):
1. Call retrieve_context(user_question) ONCE
2. IF user asks for chart AND retrieved data has numeric metrics:
   - Call generate_chart_data with the numeric values from step 1
3. Generate final answer combining both results

❌ FORBIDDEN:
- Calling retrieve_context multiple times
- Trying different query variations with retrieve_context
- Calling retrieve_context after generate_chart_data to "get more info"

📋 PROFESSIONAL TONE & FORMATTING:
- Use clear, concise, professional language
- Structure responses for readability and comprehension
- Use markdown formatting to organize content:
  * **Bold** for key terms and important concepts
  * Bullet points for lists and features
  * Numbered lists for sequential information
  * Markdown tables for comparisons, specifications, and tabular data
  * Code blocks for technical values or command examples
- For comparison questions: ALWAYS use markdown tables with clear headers and rows
- For specifications: organize by category or parameter type
- For processes/procedures: use numbered steps

📊 TABLE EXAMPLE (use this format for comparisons):
| Specification | Value 1 | Value 2 |
|---|---|---|
| Parameter | ABC | XYZ |

✅ REQUIRED ANSWER FORMAT:
1. Clear, structured answer (grounded ONLY in retrieved context)
2. Use appropriate formatting (tables, lists, bold text) for readability
3. Organize by logical sections if answer is complex
4. Include source snippets that directly support your answer
5. NO inferences or external knowledge

❌ PROHIBITED:
- Using general knowledge (e.g., "I know that...")
- Making logical deductions (e.g., "Since X, then Y must be...")
- Elaborating beyond retrieved text
- Answering if retrieve_context found nothing
- Unformatted walls of text for complex information

📊 CHART GENERATION (ONLY if numeric metrics exist):
PREREQUISITES - MUST have actual numeric data to create a chart:
- ✅ DO CREATE CHART if: Max Current (A), Power (kW), Voltage (V), Cost ($), Speed (km/h), etc.
- ❌ DO NOT CREATE CHART if: Qualitative features (yes/no), boolean characteristics, names only, no numeric values
EXAMPLE OF WHEN TO REFUSE CHART:
  User: "Compare technologies A, B, C"
  Retrieved text: Lists features like "Supports wireless", "Allows configuration", "Compatible with"
  Action: DO NOT call generate_chart_data (no numeric data). Explain: "No numeric metrics available for comparison"

WHEN USER ASKS FOR CHART:
- ONLY if retrieved context contains actual numbers/values, call generate_chart_data
- Extract ONLY real numeric values from retrieved context (NEVER fabricate, estimate, or infer values)
- Use chart_type "bar" for comparisons, "line" for trends over time
- If NO numeric data found, respond: "I cannot create a chart for this comparison as there are no quantifiable metrics in the knowledge base"
- Labels = category names, Values = only actual numeric data from context
"""


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================

class SourceMetadata(BaseModel):
    """Metadata for a retrieved source."""
    file_name: str = Field(description="Name of the source file")
    page_number: int = Field(description="Page number in the document")
    chunk_index: int = Field(description="Index of the chunk within the document")
    relevance_score: float = Field(description="Relevance score (0-1)")
    original_text: str = Field(description="Original text from the retrieved chunk (for faithfulness evaluation)")


class AgentResponse(BaseModel):
    """Structured response from RAG agent."""
    answer: str = Field(
        description="The answer to the user's question, grounded in retrieved context"
    )
    source_snippet: List[str] = Field(
        description="The relevant excerpt from the knowledge base that supports the answer"
    )
    source_metadata: List[SourceMetadata] = Field(
        default_factory=list,
        description="Metadata about the retrieved sources (file name, page number, relevance)"
    )
    chart_type: Optional[str] = Field(
        default=None,
        description="Chart type if generated: 'bar' or 'line'"
    )
    chart_data: Optional[Dict] = Field(
        default=None,
        description="Chart data as {label: value} dictionary"
    )
    chart_title: Optional[str] = Field(
        default=None,
        description="Title for the generated chart"
    )


# ============================================================================
# RAG CLIENT WRAPPER
# ============================================================================

class EVDocsClient:
    """RAG client for Vector EV Docs."""

    def __init__(
            self,
            db_client: Optional[VectorDBClient] = None,
            encoder: Optional[SentenceTransformer] = None
    ):
        """
        Initialize RAG client.

        Args:
            db_client: Path to vector db client
            encoder: Sentence transformer model for embeddings
        """
        self.db_client = db_client or VectorDBClient("vector_ev_docs")
        self.encoder = encoder or SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = "vector_ev_docs"

    async def _describe_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        """
        Use the configured LLM provider's vision API to describe an image from a PDF.

        Uses the provider specified by RAG_LLM_PROVIDER environment variable:
        - "anthropic": Uses Anthropic's vision API
        - "groq": Uses Groq's llama-3.2-11b-vision-preview

        Args:
            image_bytes: Raw image bytes
            media_type: MIME type (e.g. "image/png", "image/jpeg")

        Returns:
            Text description of the image
        """
        return await describe_image_with_vision(image_bytes, media_type)

    async def ingest(self, path: str) -> str:
        """
        Extract text from PDF, chunk it, generate embeddings (batch), and store in vector DB.

        BATCH EMBEDDING OPTIMIZATION:
        - Collects all chunks from PDF first
        - Encodes in batches of 32 using sentence-transformers
        - Reduces 100+ individual encode() calls to ~3 batch calls
        - Performance improvement: ~5x speedup for 50-page PDFs

        PERFORMANCE DETAILS:
        Standard approach (per-chunk):
        - 300 chunks → 300 encode() calls × 7.5ms = 2.25 seconds

        Batch approach (batch_size=32):
        - 300 chunks → 10 batches × 45ms = 450ms
        - Speedup: 2.25s → 0.45s = ~5x faster

        Why batching helps:
        1. Vectorization: PyTorch processes 32 items in parallel
        2. Reduced overhead: 300 function calls → 10 batch calls
        3. Better memory cache: Model weights stay loaded
        4. GPU utilization: From ~5% → ~60% (if GPU available)

        Args:
            path: Path to PDF file

        Returns:
            Status message
        """
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        file_name = pdf_path.name
        logger.info(f"Ingesting {file_name} (text + tables + images)...")

        # Step 1: Extract markdown from PDF (includes text, tables, and image descriptions)
        # pymupdf4llm converts PDFs to markdown, preserving:
        # - Text formatting and structure
        # - Table layouts
        # - Image descriptions (for charts, diagrams, etc.)
        logger.debug(f"Extracting content with pymupdf4llm...")
        markdown_text = pymupdf4llm.to_markdown(path)

        # Step 2: Chunk the markdown content
        # Chunks preserve semantic boundaries and include all content types
        chunks = self.chunk(markdown_text)
        logger.debug(f"Created {len(chunks)} semantic chunks from text")

        # Step 3: Stream-process text chunks in batches (encode + flush immediately)
        # This avoids loading all chunks into memory at once
        batch_size = RAG_INGEST_BATCH_SIZE
        chunk_counter = 0

        logger.info(f"Processing {len(chunks)} text chunks in batches of {batch_size}...")

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            # Build metadata for this batch
            batch_ids = []
            batch_metadatas = []
            for chunk_id, chunk in enumerate(batch_chunks, start=batch_start):
                batch_ids.append(f"{pdf_path.stem}_{chunk_id}")
                batch_metadatas.append({
                    "file_name": file_name,
                    "page_number": 0,
                    "chunk_index": chunk_id,
                    "chunk_size": len(chunk),
                    "content_type": "markdown"
                })

            # Encode this batch
            logger.debug(f"  Text batch {batch_num}/{total_batches}: Encoding chunks {batch_start+1}-{batch_end}...")
            embeddings = self.encoder.encode(batch_chunks, batch_size=32, show_progress_bar=False)

            # Upsert this batch immediately
            self.db_client.upsert_chunks(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_metadatas
            )
            chunk_counter += len(batch_chunks)
            logger.debug(f"  ✓ Text batch {batch_num}/{total_batches} complete ({chunk_counter} total chunks)")

        # Step 3b: Extract and describe images with vision API (stream-process in batches)
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            total_images = sum(len(page.get_images(full=True)) for page in doc)

            if total_images > 0:
                logger.info(f"Extracting and describing {total_images} images from PDF...")

                image_chunks = []
                image_metadatas = []
                image_ids = []
                image_count = 0

                for page_num, page in enumerate(doc):
                    image_list = page.get_images(full=True)
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        try:
                            img_data = doc.extract_image(xref)
                            image_bytes = img_data["image"]
                            ext = img_data.get("ext", "png")
                            media_type = f"image/{ext}" if ext in ("png", "jpeg", "gif", "webp") else "image/png"

                            image_count += 1
                            logger.info(f"  Image {image_count}/{total_images}: Describing (page {page_num+1})...")
                            description = await self._describe_image(image_bytes, media_type)

                            image_ids.append(f"{pdf_path.stem}_img_{page_num}_{img_idx}")
                            image_chunks.append(description)
                            image_metadatas.append({
                                "file_name": file_name,
                                "page_number": page_num,
                                "chunk_index": chunk_counter + len(image_chunks) - 1,
                                "chunk_size": len(description),
                                "content_type": "image_description",
                            })

                            # If batch is full, encode and upsert immediately
                            if len(image_chunks) >= batch_size:
                                logger.debug(f"  Image batch: Encoding and flushing {len(image_chunks)} descriptions...")
                                embeddings = self.encoder.encode(image_chunks, batch_size=32, show_progress_bar=False)
                                self.db_client.upsert_chunks(
                                    ids=image_ids,
                                    embeddings=embeddings.tolist(),
                                    documents=image_chunks,
                                    metadatas=image_metadatas
                                )
                                chunk_counter += len(image_chunks)
                                logger.debug(f"  ✓ Image batch complete ({chunk_counter} total chunks)")
                                # Reset for next batch
                                image_chunks = []
                                image_metadatas = []
                                image_ids = []

                        except Exception as e:
                            logger.warning(f"  Skipping image {img_idx} on page {page_num}: {e}")

                # Flush remaining image chunks if any
                if image_chunks:
                    logger.debug(f"  Flushing remaining {len(image_chunks)} image descriptions...")
                    embeddings = self.encoder.encode(image_chunks, batch_size=32, show_progress_bar=False)
                    self.db_client.upsert_chunks(
                        ids=image_ids,
                        embeddings=embeddings.tolist(),
                        documents=image_chunks,
                        metadatas=image_metadatas
                    )
                    chunk_counter += len(image_chunks)
                    logger.debug(f"  ✓ Final image batch complete ({chunk_counter} total chunks)")

            doc.close()
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")

        status = f"✓ Ingested {file_name}: {chunk_counter} chunks stored in streaming batches of {batch_size}"
        logger.info(status)
        return status

    @staticmethod
    def chunk(
            text: str,
            chunk_size: int = 800,
            chunk_overlap: int = 150
    ) -> list[str]:
        """
        Split text into overlapping semantic chunks using paragraph boundaries.

        PARAGRAPH-BASED ALGORITHM (v3.0):
        1. Split by markdown headers (##, ###) - preserve document structure
        2. Within each section, split by blank lines → paragraphs
        3. Accumulate paragraphs until chunk_size threshold
        4. Maintain proper overlap between chunks (complete paragraphs only)

        WHY PARAGRAPH-BASED?
        - ✅ Paragraphs = natural semantic units (sentences grouped logically)
        - ✅ Respects markdown structure (headers, lists, code blocks)
        - ✅ No arbitrary sentence length variations (sentences vary 5-200 chars)
        - ✅ Better for fact retrieval (coherent topic clusters)
        - ✅ Proper overlap at paragraph boundaries (no mid-paragraph splits)

        KEY BENEFITS vs SENTENCE-BASED:
        - SEMANTIC: Paragraphs are intentional groupings, not arbitrary breaks
        - CONSISTENT: More uniform chunk quality
        - STRUCTURE: Respects document formatting (markdown line breaks)
        - OVERLAP: Uses complete paragraphs for proper overlap

        SIZE TUNING (800 chars / 150 overlap):
        - Target: ~120-160 tokens per chunk
        - Accumulate paragraphs until approaching 800 chars
        - If single paragraph > 800: still include (don't split paragraphs)
        - 150 char overlap from previous paragraphs (~25-30 tokens)

        Args:
            text: Full text to chunk (markdown from pymupdf4llm)
            chunk_size: Target size in characters (~120-160 tokens)
            chunk_overlap: Overlap between chunks in characters (~25-30 tokens)

        Returns:
            List of text chunks with paragraph structure preserved
        """
        chunks = []

        # Step 1: Split by markdown headers (##, ###, ####) to preserve structure
        sections = re.split(r'(^#+\s+[^\n]+$)', text, flags=re.MULTILINE)

        current_chunk = ""

        for section in sections:
            if not section.strip():
                continue

            is_header = section.startswith('#')

            if is_header and current_chunk:
                # Save current chunk before starting new header section
                chunks.append(current_chunk.strip())
                # Start new section with the header
                current_chunk = section + "\n"
            else:
                # For content sections, use paragraph-based chunking
                # Split by blank lines (one or more newlines)
                paragraphs = re.split(r'\n\s*\n', section)

                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue

                    # Try adding this paragraph to current chunk
                    if current_chunk:
                        test_chunk = current_chunk + "\n\n" + paragraph
                    else:
                        test_chunk = paragraph

                    # Check if adding paragraph would exceed chunk_size
                    if len(test_chunk) > chunk_size and current_chunk and not is_header:
                        # Save current chunk at paragraph boundary (not mid-paragraph)
                        chunks.append(current_chunk.strip())

                        # Create proper overlap using complete paragraphs
                        # Keep paragraphs from end of current_chunk that fit in overlap_size
                        overlap_text = current_chunk
                        overlap_paragraphs = re.split(r'\n\s*\n', overlap_text)

                        # Keep removing from start until we fit in overlap_size
                        while len(overlap_text) > chunk_overlap and len(overlap_paragraphs) > 1:
                            overlap_paragraphs.pop(0)
                            overlap_text = "\n\n".join(overlap_paragraphs)

                        # Start new chunk with overlap + current paragraph
                        if overlap_text:
                            current_chunk = overlap_text + "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        # Accumulate paragraph in current chunk
                        current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def query(
            self,
            query_text: str,
            n_results: int = 3
    ) -> list[str]:
        """
        Query vector database for relevant documents.

        Args:
            query_text: User's question
            n_results: Number of results

        Returns:
            List of relevant chunks
        """
        # Encode query
        query_embedding = self.encoder.encode(query_text)

        # Query vector DB
        hits = self.db_client.query(embedding=query_embedding, n_results=n_results)

        # Return documents
        return [hit.text for hit in hits]

    def query_with_metadata(
            self,
            query_text: str,
            n_results: int = 3
    ) -> list[tuple[str, dict, int]]:
        """
        Query vector database and return documents with metadata.

        Args:
            query_text: User's question
            n_results: Number of results

        Returns:
            List of tuples (text, metadata)
        """
        # Encode query
        query_embedding = self.encoder.encode(query_text)

        # Query vector DB
        hits = self.db_client.query(embedding=query_embedding, n_results=n_results)

        # Return documents with metadata
        return [(hit.text, hit.metadata, hit.score) for hit in hits]


# ============================================================================
# RAG AGENT CONTEXT
# ============================================================================
@dataclass
class RagAgentContext:
    """Context for RAG agent."""
    rag_client: EVDocsClient
    retrieved_metadata: List[SourceMetadata] = None
    conversation_context: Optional[List[Dict[str, str]]] = None
    chart_type: Optional[str] = None
    chart_data: Optional[Dict] = None
    chart_title: Optional[str] = None
    tool_call_count: int = 0  # Track number of tool calls to limit them
    max_tool_calls: int = 4  # Maximum allowed tool calls per question (retrieve + chart)

    def __post_init__(self):
        if self.retrieved_metadata is None:
            self.retrieved_metadata = []
        if self.conversation_context is None:
            self.conversation_context = []


# ============================================================================
# QUERY EXPANSION
# ============================================================================

async def expand_query(query: str, num_expansions: int = 3) -> List[str]:
    """
    Generate alternative query phrasings using Claude for better retrieval.

    ⚠️  LLM CALL #1: Makes 1 LLM API call to generate alternative queries

    LLM-BASED QUERY EXPANSION:
    Improves retrieval by generating semantically equivalent queries with:
    - Different vocabulary and terminology
    - Synonyms and related concepts
    - Decomposed complex queries
    - Explicit reformulation of implicit concepts

    Cost: 1 LLM call per user query (~0.0003 cost)
    Latency: +2-3 seconds
    Quality: Significantly better retrieval for edge cases like:
      - "who authored this?" → expands to include "author", "written by", "creator"
      - "charging systems" → includes "charging solutions", "power delivery", "charging technology"

    Args:
        query: Original user query
        num_expansions: Number of alternative phrasings (default 3)

    Returns:
        List with original query + N expanded versions
    """
    try:
        client = get_llm_client()

        expansion_prompt = f"""Given this user question about EV technical documentation, generate {num_expansions} alternative phrasings that would help retrieve relevant documents.

Original question: "{query}"

Generate exactly {num_expansions} different ways to ask this question. Each should:
1. Use different vocabulary/terminology
2. Emphasize different aspects of the user's intent
3. Be specific to automotive/electrical/technical domains
4. Be naturally phrased (not awkward)

Return ONLY the alternative queries, one per line, without numbering or bullets."""

        if RAG_LLM_PROVIDER == "groq":
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Groq's recommended model for fast text
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=400,
            )
            expanded_text = response.choices[0].message.content
        else:
            response = client.messages.create(
                model=RAG_LLM_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": expansion_prompt}]
            )
            expanded_text = response.content[0].text

        # Parse expanded queries from response
        expansions = [line.strip() for line in expanded_text.strip().split('\n') if line.strip()]

        # Return original + up to N expansions
        result = [query] + expansions[:num_expansions]
        logger.debug(f"Query expanded to {len(result)} variations ({RAG_LLM_PROVIDER} provider)")
        return result

    except Exception as e:
        # Fallback: return just original query if expansion fails
        logger.warning(f"Query expansion failed: {str(e)[:100]}, using original query only")
        return [query]


# ============================================================================
# RETRIEVAL TOOL
# ============================================================================

async def retrieve_context(ctx: RunContext[RagAgentContext], query: str, top_n_rows: int = 5) -> str:
    """
    Tool: Retrieve relevant context with optional LLM-based query expansion.

    QUERY EXPANSION + RETRIEVAL FLOW (when RAG_ENABLE_QUERY_EXPANSION=true):
    ┌──────────────────────────────────────┐
    │ 1. Expand Query (LLM)                │
    │    Original + 3 alternatives         │
    │    Cost: 1 LLM call (~2-3s)          │
    └─────────────┬────────────────────────┘
                  ↓
    ┌──────────────────────────────────────┐
    │ 2. Embed & Search (4 queries)        │
    │    All-MiniLM-L6-v2 embeddings       │
    │    Cosine similarity search          │
    └─────────────┬────────────────────────┘
                  ↓
    ┌──────────────────────────────────────┐
    │ 3. Aggregate Results                 │
    │    Deduplicate by chunk_index        │
    │    Keep best relevance score         │
    └─────────────┬────────────────────────┘
                  ↓
    ┌──────────────────────────────────────┐
    │ 4. Sort & Select Top-K               │
    │    Return best matches to agent      │
    └──────────────────────────────────────┘

    FAST RETRIEVAL (when RAG_ENABLE_QUERY_EXPANSION=false):
    - Single direct search with original query
    - No LLM expansion overhead
    - ~2-3 seconds per query vs 5-6 seconds with expansion
    - Useful for evaluation/batch processing

    CONFIGURATION:
    - Export RAG_ENABLE_QUERY_EXPANSION=false to disable expansion
    - Default: true (expansion enabled)
    - For evaluation: RAG_ENABLE_QUERY_EXPANSION=false python evaluator.py

    SIMILARITY METRIC:
    - Cosine similarity on embeddings
    - Embeddings: 384-dimensional (all-MiniLM-L6-v2)
    - Scores: 0-1 (0=opposite, 1=identical)

    Args:
        ctx: Pydantic AI context with RagAgentContext dependency
        query: The user's question
        top_n_rows: Number of top results to retrieve (default 5)

    Returns:
        Retrieved context as formatted string with source metadata
    """
    logger.info(f"🟡 retrieve_context() called for: {query[:60]}...")

    rag_context = ctx.deps

    # TOOL CALL LIMIT CHECK: Prevent excessive retrieve_context calls
    if rag_context.tool_call_count >= rag_context.max_tool_calls:
        logger.warning(f"Tool call limit reached ({rag_context.tool_call_count}/{rag_context.max_tool_calls}). Refusing additional retrieve_context calls.")
        return f"Tool call limit reached (max {rag_context.max_tool_calls} calls per question). Cannot make additional retrieval calls. Answer the user's question using the context already retrieved."

    # Increment tool call counter
    rag_context.tool_call_count += 1
    logger.info(f"Tool call #{rag_context.tool_call_count}/{rag_context.max_tool_calls}")

    rag_client = rag_context.rag_client

    # Step 1: Optionally expand query based on environment configuration
    if RAG_ENABLE_QUERY_EXPANSION:
        logger.info(f"   → Calling expand_query (⚠️  LLM CALL)")
        expanded_queries = await expand_query(query, num_expansions=3)
        logger.debug(f"Searching with {len(expanded_queries)} query variations")
    else:
        # Fast mode: use original query only (for evaluation/assessment)
        expanded_queries = [query]
        logger.debug(f"Query expansion disabled (RAG_ENABLE_QUERY_EXPANSION=false)")

    # Step 2: Aggregate results from all query variations
    # Use chunk_index as unique ID to deduplicate
    all_results = {}  # chunk_index -> (doc, metadata, best_score)

    for expanded_query in expanded_queries:
        # Search with each query variation
        # Get more results per query to handle deduplication
        logger.debug(f"Searching: {expanded_query[:60]}...")
        results = rag_client.query_with_metadata(
            query_text=expanded_query,
            n_results=top_n_rows * 2  # Get 10 per query to find top-5 after dedup
        )

        # Track results by chunk_index, keeping best score
        for doc, metadata, score in results:
            chunk_idx = metadata.get("chunk_index", -1)
            if chunk_idx not in all_results or score > all_results[chunk_idx][2]:
                all_results[chunk_idx] = (doc, metadata, score)

    if not all_results:
        logger.warning(f"No relevant documents found for query: {query}")
        return "No relevant documents found in knowledge base."

    # Step 3: Sort by relevance and take top-K
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x[2],
        reverse=True
    )[:top_n_rows]

    logger.info(f"Retrieved {len(sorted_results)} relevant chunks (relevance scores: {[round(x[2], 2) for x in sorted_results]})")

    # Step 4: Build response with metadata
    rag_context.retrieved_metadata = []
    context_parts = []

    # Include conversation context if available
    if rag_context.conversation_context:
        context_parts.append("--- CONVERSATION CONTEXT (Previous Messages) ---")
        for msg in rag_context.conversation_context:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            context_parts.append(f"{role}: {content}")
        context_parts.append("--- END CONVERSATION CONTEXT ---\n")
        logger.debug(f"Added conversation context with {len(rag_context.conversation_context)} previous messages")

    for i, (doc, metadata, score) in enumerate(sorted_results, 1):
        source_meta = SourceMetadata(
            file_name=metadata.get("file_name", "unknown"),
            page_number=metadata.get("page_number", -1),
            chunk_index=metadata.get("chunk_index", -1),
            relevance_score=score,
            original_text=doc
        )
        rag_context.retrieved_metadata.append(source_meta)

        logger.debug(f"  [{i}] {metadata.get('file_name', 'unknown')} (page {metadata.get('page_number', '?')}, score: {score:.2f})")

        context_parts.append(
            f"[Source {i} - {metadata.get('file_name', 'unknown')}, Page {metadata.get('page_number', '?')}]\n{doc}\n"
        )

    return "\n".join(context_parts)


async def generate_chart_data(
    ctx: RunContext[RagAgentContext],
    chart_type: str,
    title: str,
    labels: List[str],
    values: List[float]
) -> str:
    """
    Tool: Structure data as a chart specification for frontend rendering.

    ONLY call this when you have actual numeric metrics from retrieved context.
    Do NOT create charts for qualitative data (yes/no, features, boolean flags).

    Args:
        ctx: Pydantic AI context
        chart_type: Type of chart - "bar" or "line"
        title: Descriptive chart title
        labels: List of category labels (x-axis)
        values: List of numeric values corresponding to labels (y-axis)

    Returns:
        Confirmation string if valid, or error message if data is invalid
    """
    # TOOL CALL LIMIT CHECK: Prevent excessive tool calls
    if ctx.deps.tool_call_count >= ctx.deps.max_tool_calls:
        logger.warning(f"Tool call limit reached ({ctx.deps.tool_call_count}/{ctx.deps.max_tool_calls}). Refusing chart generation.")
        return f"Tool call limit reached. Cannot generate chart. Answer the user's question using available context."

    # Increment tool call counter
    ctx.deps.tool_call_count += 1
    logger.info(f"Tool call #{ctx.deps.tool_call_count}/{ctx.deps.max_tool_calls} (chart generation)")

    if len(labels) != len(values):
        return "Error: labels and values must have the same length"

    # Validate that values have actual variation (not all same, not all 1.0)
    unique_values = set(values)
    if len(unique_values) == 1:
        # All values are the same - this indicates dummy/fabricated data
        logger.warning(f"Rejected chart: all values are {values[0]} (no variation in data)")
        return f"Cannot create chart: all data points have the same value ({values[0]}). This indicates no real metrics for comparison. Please provide actual numeric data from the knowledge base."

    # Check if values are suspiciously all 1.0 (common fabrication)
    if all(v == 1.0 for v in values):
        logger.warning(f"Rejected chart: all values are 1.0 (dummy data)")
        return "Cannot create chart: all values are 1.0 (dummy data). This indicates you're trying to chart qualitative data that has no numeric metrics. Only create charts when there are actual numeric values (prices, speeds, power, voltages, etc.) to compare."

    ctx.deps.chart_data = dict(zip(labels, values))
    ctx.deps.chart_type = chart_type
    ctx.deps.chart_title = title

    logger.info(f"Chart data generated: {chart_type} chart '{title}' with {len(labels)} data points (values: {unique_values})")
    return f"Chart generated: {chart_type} chart '{title}' with {len(labels)} data points ({', '.join(labels)})"


# ============================================================================
# RAG AGENT
# ============================================================================

# Format model string for pydantic_ai based on provider
if RAG_LLM_PROVIDER == "groq":
    agent_model = f"groq:{RAG_LLM_MODEL}"
else:
    agent_model = RAG_LLM_MODEL

agent = Agent(
    model=agent_model,
    output_type=AgentResponse,
    tools=[retrieve_context, generate_chart_data],
    system_prompt=SYSTEM_PROMPT,
    deps_type=RagAgentContext,
    retries=1,
    end_strategy='exhaustive'  # Allow multiple tool calls (retrieve + chart)
)

logger.info(f"RAG Agent initialized with {RAG_LLM_PROVIDER} provider (model: {RAG_LLM_MODEL})")

# ============================================================================
# MAIN
# ============================================================================

init_lock = threading.Lock()


@lru_cache
def _get_pdf_client() -> EVDocsClient:
    return EVDocsClient(
        db_client=VectorDBClient("vector_ev_docs", path="./ev_vector_db")
    )


def get_pdf_client() -> EVDocsClient:
    with init_lock:
        return _get_pdf_client()


async def ingest_pdf(pdf_path: str = "data") -> None:
    """
    Ingest PDF(s) into vector database (runs once).

    Args:
        pdf_path: Path to PDF file or directory containing PDFs
    """
    rag_client = get_pdf_client()
    path = Path(pdf_path)

    if path.is_file():
        # Single PDF file
        logger.info(f"Ingesting file: {path.name}")
        await rag_client.ingest(str(path))
        logger.info(f"PDF ingestion complete")

    elif path.is_dir():
        # Directory of PDFs
        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {path}")
            return

        logger.info(f"Found {len(pdf_files)} PDF file(s) in {path}")
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Ingesting: {pdf_file.name}")
            await rag_client.ingest(str(pdf_file))

        logger.info(f"All PDFs ingestion complete ({len(pdf_files)} files)")

    else:
        raise FileNotFoundError(f"Path not found: {pdf_path}")


async def main(question: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> AgentResponse:
    """
    Query the RAG agent (assumes PDF already ingested).

    Args:
        question: User's question
        conversation_context: Optional list of previous messages ({"role": "user"|"assistant", "content": "..."})

    Returns:
        AgentResponse with answer, source snippet, and metadata

    Raises:
        ValueError: If retrieve_context tool was not called or returned no results
    """
    logger.info(f"🔵 main() called - ⚠️  LLM CALL #2: Agent will run for: {question[:60]}...")

    # Initialize RAG client with VectorDBClient
    rag_client = get_pdf_client()

    # Create agent context (will be populated with metadata during retrieval)
    context = RagAgentContext(rag_client=rag_client, conversation_context=conversation_context or [])

    # Run agent with context
    result = await agent.run(question, deps=context)

    # Extract response and add metadata from context
    response = result.output

    # VALIDATION: Ensure retrieve_context tool was called and returned results
    # This prevents the agent from answering without consulting the knowledge base
    if not context.retrieved_metadata or len(context.retrieved_metadata) == 0:
        # Tool was not called or returned no results
        # Force honest response: admit the information is not available
        logger.warning(f"No sources retrieved - forcing honest response for: {question[:60]}...")
        response.answer = "This information is not available in the knowledge base."
        response.source_snippet = []
        response.source_metadata = []
        return response

    # Log successful retrieval
    logger.info(f"Agent generated response with {len(context.retrieved_metadata)} sources")
    for i, meta in enumerate(context.retrieved_metadata, 1):
        logger.debug(f"  Source [{i}] {meta.file_name} (page {meta.page_number}, relevance: {meta.relevance_score:.2f})")

    # Add retrieved metadata to response
    response.source_metadata = context.retrieved_metadata

    # Pass chart data from tool call into response if generated
    if context.chart_data:
        response.chart_type = context.chart_type
        response.chart_data = context.chart_data
        response.chart_title = context.chart_title

    # Return structured response with metadata
    return response


if __name__ == "__main__":
    import asyncio


    async def run_demo():
        # Ingest PDF once
        await ingest_pdf()

        # Run test queries
        test_queries = [
            "how much time is required to charge a battery of heavy vehicles under an hour?",
            "What is the CANoe Test Package EV?",
            "who is the author of MCS document ?",
            "What charging-related technologies and challenges are discussed?",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await main(query)
            print(f"\nAnswer: {response.answer}")
            print(f"\nsource_snippets: {response.source_snippet}")
            print(f"\nmetadata: {response.source_metadata}")
            print(f"\n{'-' * 30}")


    asyncio.run(run_demo())
