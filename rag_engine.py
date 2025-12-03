# rag_engine.py
"""
SQLite-backed MiniRAG engine with:
- persistent embeddings stored in db/embeddings.db
- incremental indexing (re-index only changed/new files)
- in-memory cache for fast similarity computations
- query caching to avoid re-embedding identical queries
"""

import os
import sqlite3
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

CHUNK_SIZE = 150  # words per chunk

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

class MiniRAG:
    def __init__(self, data_folder: str = "data", db_path: str = "db/embeddings.db"):
        self.data_folder = data_folder
        self.db_path = db_path

        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # load embedding model once
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # connect sqlite
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

        # in-memory stores
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.embeddings: np.ndarray = np.zeros(
            (0, self.model.get_sentence_embedding_dimension()), dtype=np.float32
        )

        # query cache
        self.query_cache = {}

        # build/update index
        self._sync_index_with_files()

    # -------------------------
    # DB schema
    # -------------------------
    def _create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                chunk TEXT,
                embedding BLOB,
                chunk_hash TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS files_meta (
                file TEXT PRIMARY KEY,
                sha1 TEXT
            )
        """)
        self.conn.commit()

    # -------------------------
    # Utils
    # -------------------------
    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE):
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    # -------------------------
    # Indexing / Sync
    # -------------------------
    def _index_file(self, filepath: str, filename: str):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self._chunk_text(text)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            emb = self.model.encode(chunk).astype(np.float32)
            emb_bytes = emb.tobytes()
            chunk_hash = _sha1(chunk)

            self.cursor.execute("""
                INSERT INTO documents (source, chunk, embedding, chunk_hash)
                VALUES (?, ?, ?, ?)
            """, (filename, chunk, emb_bytes, chunk_hash))

        self.conn.commit()

    def _sync_index_with_files(self):
        files = [
            f for f in os.listdir(self.data_folder)
            if f.lower().endswith((".txt", ".md"))
        ]

        # compute current sha1
        current_meta = {}
        for fname in files:
            path = os.path.join(self.data_folder, fname)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            current_meta[fname] = _sha1(content)

        # load stored meta
        self.cursor.execute("SELECT file, sha1 FROM files_meta")
        stored = dict(self.cursor.fetchall())

        # add new or updated
        for fname, sha in current_meta.items():
            if fname not in stored:
                print(f"Indexing new file: {fname}")
                self._index_file(os.path.join(self.data_folder, fname), fname)
                self.cursor.execute(
                    "INSERT OR REPLACE INTO files_meta (file, sha1) VALUES (?, ?)",
                    (fname, sha)
                )
                self.conn.commit()
            elif stored.get(fname) != sha:
                print(f"File updated, re-indexing: {fname}")
                self.cursor.execute("DELETE FROM documents WHERE source = ?", (fname,))
                self.conn.commit()
                self._index_file(os.path.join(self.data_folder, fname), fname)
                self.cursor.execute(
                    "UPDATE files_meta SET sha1=? WHERE file=?",
                    (sha, fname)
                )
                self.conn.commit()

        # delete removed
        removed = set(stored.keys()) - set(current_meta.keys())
        for fname in removed:
            print(f"File removed: {fname}, deleting its entries.")
            self.cursor.execute("DELETE FROM documents WHERE source=?", (fname,))
            self.cursor.execute("DELETE FROM files_meta WHERE file=?", (fname,))
            self.conn.commit()

        self._load_from_db()

    def _load_from_db(self):
        self.cursor.execute("SELECT source, chunk, embedding FROM documents")
        rows = self.cursor.fetchall()

        chunks = []
        sources = []
        emb_list = []
        for src, chunk, emb_blob in rows:
            chunks.append(chunk)
            sources.append(src)
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            emb_list.append(emb)

        self.chunks = chunks
        self.sources = sources

        if len(emb_list) > 0:
            self.embeddings = np.vstack(emb_list)
        else:
            self.embeddings = np.zeros(
                (0, self.model.get_sentence_embedding_dimension()),
                dtype=np.float32
            )

        print(f"Loaded {len(self.chunks)} chunks from embeddings DB.")

    # -------------------------
    # Query
    # -------------------------
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def query(self, question: str, k: int = 3):
        q_hash = self._hash(question)
        if q_hash in self.query_cache:
            return self.query_cache[q_hash]

        if len(self.chunks) == 0:
            return []

        q_emb = self.model.encode([question]).astype(np.float32)[0]

        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-12)
        sims = np.dot(self.embeddings, q_emb) / norms

        idxs = sims.argsort()[-k:][::-1]
        results = []
        for idx in idxs:
            results.append({
                "chunk": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(sims[idx])
            })

        self.query_cache[q_hash] = results
        return results

    def clear_query_cache(self):
        self.query_cache = {}
