"""Index data/source.md into Qdrant chunks_collection (idempotent)."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "source.md"


def detect_section(chunk_text: str, fallback: str) -> str:
    for line in chunk_text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            return line[3:].strip()
    return fallback


def main() -> int:
    load_dotenv(ROOT / ".env")

    url = os.environ["QDRANT_URL"]
    key = os.environ["QDRANT_API_KEY"]
    collection = os.environ.get("QDRANT_CHUNKS_COLLECTION", "chunks_collection")
    chunk_size = int(os.environ.get("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "50"))
    embed_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_name = embed_model.split("/", 1)[-1]

    if not SOURCE.exists():
        print(f"[index] ERROR: source missing → run scripts/download_doc.py first ({SOURCE})", file=sys.stderr)
        return 1

    text = SOURCE.read_text(encoding="utf-8")
    print(f"[index] loaded {len(text)} chars from {SOURCE.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f"[index] split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    print(f"[index] loading embedder {embed_name} ...")
    embedder = SentenceTransformer(embed_name)
    vectors = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    print(f"[index] encoded {len(vectors)} vectors (dim={vectors.shape[1]})")

    qdrant = QdrantClient(url=url, api_key=key, timeout=60)
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
        print(f"[index] dropped existing collection '{collection}'")
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"[index] created collection '{collection}' (384-dim, cosine)")

    points: list[PointStruct] = []
    section_carry = "Intro"
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        section = detect_section(chunk, section_carry)
        section_carry = section
        chunk_id = f"chunk_{i:04d}"
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "section": section,
                    "source_path": "data/source.md",
                },
            )
        )

    BATCH = 64
    for i in range(0, len(points), BATCH):
        qdrant.upsert(collection_name=collection, points=points[i : i + BATCH])
    print(f"[index] upserted {len(points)} points into '{collection}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
