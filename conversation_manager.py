"""
Conversation Manager: SQLite persistence + in-memory cache for chat history.

Stores conversations in SQLite for persistence, maintains last 2-3 conversations
in memory for fast access. Handles session management and message history.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history with SQLite persistence and in-memory cache."""

    def __init__(self, db_path: str = "./conversations.db", max_cache_size: int = 3):
        """
        Initialize conversation manager.

        Args:
            db_path: Path to SQLite database file
            max_cache_size: Number of conversations to keep in memory (default 3)
        """
        self.db_path = db_path
        self.max_cache_size = max_cache_size

        # In-memory cache: conversation_id -> conversation data
        self.memory_cache = OrderedDict()

        # Initialize database
        self._init_db()

        logger.debug(f"ConversationManager initialized (cache size: {max_cache_size})")

    def _init_db(self) -> None:
        """Create SQLite tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
            """
        )

        # Messages table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)"
        )

        conn.commit()
        conn.close()

        logger.debug("SQLite database initialized")

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: str = "New Conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new conversation.

        Args:
            conversation_id: Unique conversation ID
            user_id: User creating the conversation
            title: Conversation title
            metadata: Optional metadata dictionary

        Returns:
            Conversation data
        """
        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "messages": [],
        }

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (id, user_id, title, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, user_id, title, json.dumps(metadata or {})),
        )
        conn.commit()
        conn.close()

        # Add to memory cache
        self.memory_cache[conversation_id] = conversation
        self._manage_cache()

        logger.info(f"Created conversation: {conversation_id}")
        return conversation

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata

        Returns:
            Message data with timestamp
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, json.dumps(metadata or {})),
        )

        # Update conversation timestamp
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )

        conn.commit()
        conn.close()

        # Update memory cache if conversation is cached
        if conversation_id in self.memory_cache:
            self.memory_cache[conversation_id]["messages"].append(message)
            self.memory_cache[conversation_id]["updated_at"] = message["timestamp"]

        logger.debug(f"Added message to conversation: {conversation_id}")
        return message

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.

        Checks memory cache first, then database.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Conversation data with all messages, or None if not found
        """
        # Check memory cache first (fast)
        if conversation_id in self.memory_cache:
            logger.debug(f"Retrieved conversation from cache: {conversation_id}")
            return self.memory_cache[conversation_id]

        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get conversation metadata
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        conv_row = cursor.fetchone()

        if not conv_row:
            conn.close()
            return None

        # Get all messages
        cursor.execute(
            "SELECT role, content, timestamp, metadata FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,),
        )
        msg_rows = cursor.fetchall()
        conn.close()

        # Build conversation object
        conversation = {
            "id": conv_row[0],
            "user_id": conv_row[1],
            "title": conv_row[2],
            "created_at": conv_row[3],
            "updated_at": conv_row[4],
            "metadata": json.loads(conv_row[5]) if conv_row[5] else {},
            "messages": [
                {
                    "role": msg[0],
                    "content": msg[1],
                    "timestamp": msg[2],
                    "metadata": json.loads(msg[3]) if msg[3] else {},
                }
                for msg in msg_rows
            ],
        }

        # Add to memory cache
        self.memory_cache[conversation_id] = conversation
        self._manage_cache()

        logger.debug(f"Retrieved conversation from database: {conversation_id}")
        return conversation

    def list_conversations(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List conversations for a user.

        Args:
            user_id: User ID
            limit: Maximum number of conversations to return

        Returns:
            List of conversations (without messages)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, metadata
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )

        conversations = [
            {
                "id": row[0],
                "user_id": row[1],
                "title": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        logger.debug(f"Listed {len(conversations)} conversations for user: {user_id}")
        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.

        Args:
            conversation_id: ID of the conversation

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if exists
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            return False

        # Delete messages first (foreign key)
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

        # Delete conversation
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

        conn.commit()
        conn.close()

        # Remove from cache
        if conversation_id in self.memory_cache:
            del self.memory_cache[conversation_id]

        logger.info(f"Deleted conversation: {conversation_id}")
        return True

    def get_context(self, conversation_id: str, last_n: int = 5) -> List[Dict[str, str]]:
        """
        Get last N messages from a conversation for RAG context.

        Args:
            conversation_id: ID of the conversation
            last_n: Number of recent messages to return

        Returns:
            List of messages in format: [{"role": "user", "content": "..."}, ...]
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        messages = conversation["messages"]
        # Get last N messages
        return [{"role": m["role"], "content": m["content"]} for m in messages[-last_n:]]

    def _manage_cache(self) -> None:
        """Remove oldest conversation if cache exceeds max size."""
        while len(self.memory_cache) > self.max_cache_size:
            oldest_id = next(iter(self.memory_cache))
            del self.memory_cache[oldest_id]
            logger.debug(f"Evicted conversation from cache: {oldest_id}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.memory_cache),
            "max_cache_size": self.max_cache_size,
            "cached_conversations": list(self.memory_cache.keys()),
        }
