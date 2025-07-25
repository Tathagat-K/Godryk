# backend/src/services/milvus_client.py

import threading, logging
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType
)
from src.core.config import settings

_logger = logging.getLogger(__name__)

class MilvusClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._connect()
            return cls._instance

    def _connect(self):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=10
        )
        _logger.info(f"✅ Connected to Milvus @ {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    def get_collection(self, name: str | None = None) -> Collection:
        name = name or settings.MILVUS_COLLECTION_NAME

        if not utility.has_collection(name):
            _logger.info(f"🚀 Creating Milvus collection '{name}'")

            # 1) Concatenated text field for BM25 (name + description)
            text_field = FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_bm25=True,
                analyzer_params={"type": "standard"}
            )

            # 2) Dense embedding
            dense_field = FieldSchema(
                name="dense",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.EMBEDDING_DIM
            )

            # 3) Sparse embedding
            sparse_field = FieldSchema(
                name="sparse",
                dtype=DataType.SPARSE_FLOAT_VECTOR
            )

            # 4) Full-row metadata JSON
            meta_field = FieldSchema(
                name="metadata",
                dtype=DataType.JSON
            )

            fields = [
                # auto-generated primary key
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                text_field,
                dense_field,
                sparse_field,
                meta_field,
            ]

            schema = CollectionSchema(
                fields,
                enable_dynamic_field=False,
                description="Hybrid collection for Sheet1.csv items"
            )

            collection = Collection(
                name=name,
                schema=schema,
                using="default"
            )

            # build indexes
            collection.create_index(
                field_name="dense",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 256}
                }
            )
            collection.create_index(
                field_name="sparse",
                index_params={
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                    "params": {"drop_ratio_build": 0.1}
                }
            )

        else:
            _logger.info(f"🔍 Loading existing Milvus collection '{name}'")
            collection = Collection(name=name, using="default")

        # Load into memory (also builds BM25 index on `text` automatically)
        collection.load()
        return collection
