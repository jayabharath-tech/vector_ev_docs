"""Vector Database client"""
from dataclasses import dataclass
from typing import Optional, Sequence, List, Any, Dict

import chromadb


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class VectorDBClient:
    def __init__(self, collection_name: str, path: Optional[str] = None):
        self.client = chromadb.PersistentClient(path) if path else chromadb.EphemeralClient()
        self.collection_name = collection_name
        self.collection = self.get_or_create_collection()

    def get_or_create_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def reset_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:  # pylint: disable=broad-except
            pass
        finally:
            self.get_or_create_collection()

    def query(self, embedding: Sequence[float], n_results: int = 5):
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        ids = res["ids"][0]
        assert res["documents"]
        docs = res["documents"][0]
        assert res["distances"]
        dists = res["distances"][0]
        assert res["metadatas"]
        metas = res["metadatas"][0]

        hits: list[Hit] = []
        for _id, doc, dist, meta in zip(ids, docs, dists, metas):
            hits.append(
                Hit(
                    id=_id,
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    text=doc,
                    metadata=meta or {},  # pyright: ignore[reportArgumentType]
                )
            )
        return hits

    def upsert_chunks(
            self,
            ids: List[str],
            embeddings: List[Sequence[float]],
            documents: List[str],
            metadatas: List[dict[str, Any]],
    ) -> None:
        self.collection.upsert(
            ids=ids,  # pyright: ignore[reportArgumentType]
            embeddings=embeddings,  # pyright: ignore[reportArgumentType]
            documents=documents,  # pyright: ignore[reportArgumentType]
            metadatas=metadatas,  # pyright: ignore[reportArgumentType]
        )
