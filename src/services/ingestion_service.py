# backend/src/services/ingestion_service.py

import io
import json
import logging
from typing import List, Tuple, Dict

import chardet
import pandas as pd
import fitz              # PyMuPDF
from docx import Document
from fastapi import HTTPException, UploadFile

from src.core.config import settings
# Avoid expensive model loading at import time
def _get_embedding_models():
    from src.services.embeddings import embedding_model, sparse_embedding_model
    return embedding_model, sparse_embedding_model
from src.db.milvus_client import MilvusClient

logger = logging.getLogger("ingestion")

CHUNK_SIZE = settings.INGEST_CHUNK_SIZE
CHUNK_OVERLAP = settings.INGEST_CHUNK_OVERLAP
ALLOWED_EXT = set(settings.ALLOWED_FILE_EXTENSIONS)

def _detect_encoding(data: bytes) -> str:
    enc = chardet.detect(data or b"")["encoding"] or "utf-8"
    logger.debug(f"Detected file encoding: {enc}")
    return enc

def _sliding_window(text: str) -> List[Tuple[str, Dict]]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        segment = words[i : i + CHUNK_SIZE]
        meta = {"chunk_start": i, "chunk_end": i + len(segment)}
        chunks.append((" ".join(segment), meta))
    return chunks

def _process_tabular(df: pd.DataFrame, source: str) -> List[Tuple[str, Dict]]:
    chunks = []
    for idx, row in df.iterrows():
        text = " | ".join(map(str, row.values))
        meta = {"row_index": idx, "source": source}
        chunks.append((text, meta))
    return chunks

def ingest_bytes(data: bytes, filename: str, user_id: str) -> None:
    logger.debug("Starting ingestion for %s by %s", filename, user_id)
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'")

    if not data:
        raise HTTPException(400, f"Empty upload for '{filename}'")

    # --- 1) Chunking by file type ---
    if ext == "txt":
        txt = data.decode(_detect_encoding(data), errors="replace")
        raw = _sliding_window(txt)

    elif ext == "pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        txt = "\n".join(p.get_text("text") for p in doc)
        raw = _sliding_window(txt)

    elif ext == "docx":
        doc = Document(io.BytesIO(data))
        txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        raw = _sliding_window(txt)

    elif ext in {"csv", "xlsx"}:
        buf = io.BytesIO(data)
        if ext == "csv":
            df = pd.read_csv(buf, on_bad_lines="skip", dtype=str).fillna("")
        else:
            df = pd.read_excel(buf, dtype=str).fillna("")
        raw = _process_tabular(df, filename)

    elif ext == "json":
        records = json.loads(data)
        if not isinstance(records, list):
            records = [records]
        raw = [(json.dumps(r), {"source": filename}) for r in records]

    else:
        raw = []

    if not raw:
        logger.warning(f"No content to ingest for '{filename}'")
        return

    texts, metas = zip(*raw)
    count = len(texts)
    logger.info(f"Embedding {count} chunks from '{filename}'")

    # --- 2) Compute embeddings ---
    embedding_model, sparse_embedding_model = _get_embedding_models()
    dense_vecs = embedding_model.encode(list(texts), show_progress_bar=False).tolist()
    sparse_mat = sparse_embedding_model.encode_documents(list(texts))
    logger.debug("Computed dense and sparse embeddings")

    def _csr_to_milvus(csr, i):
        s, e = csr.indptr[i], csr.indptr[i+1]
        return {int(dim): float(val) for dim, val in zip(csr.indices[s:e], csr.data[s:e])}

    sparse_vecs = [_csr_to_milvus(sparse_mat, i) for i in range(count)]

    # annotate metadata
    for m in metas:
        m.setdefault("uploader_id", user_id)

    # --- 3) Insert into Milvus ---
    client = MilvusClient()
    coll = client.get_collection()
    # ensure partition per extension
    if not coll.has_partition(ext):
        coll.create_partition(ext)
    logger.info(f"Inserting into partition '{ext}' ({count} vectors)")

    coll.insert(
        [
            list(texts),     # text (for BM25)
            dense_vecs,      # dense embeddings
            sparse_vecs,     # sparse TF-IDF embeddings
            list(metas),     # full metadata JSON
        ],
        partition_name=ext
    )
    logger.debug("Inserted %d records into Milvus", count)
    logger.info(f"✅ Completed ingestion for '{filename}'")

async def ingest_file(upload: UploadFile, user_id: str) -> None:
    logger.debug("Reading upload %s", upload.filename)
    data = await upload.read()
    ingest_bytes(data, upload.filename, user_id)
