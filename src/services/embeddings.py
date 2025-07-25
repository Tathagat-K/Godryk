# backend/src/services/embeddings.py

import logging
from sentence_transformers import SentenceTransformer
from pymilvus import model  # <— brings in the model subpackage
from src.core.config import settings

logger = logging.getLogger(__name__)

# Dense embedder
try:
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    logger.info(f"✅ Loaded dense embedding model: {settings.EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to load dense embedding model: {e}")
    raise

# Sparse embedder (SPLADE via Milvus’s model extra)
try:
    sparse_embedding_model = model.sparse.SpladeEmbeddingFunction(
        model_name=settings.SPARSE_EMBEDDING_MODEL_NAME,
        device=settings.SPARSE_EMBEDDING_DEVICE,
    )
    logger.info(f"✅ Loaded sparse embedding model: {settings.SPARSE_EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to load sparse embedding model: {e}")
    raise
