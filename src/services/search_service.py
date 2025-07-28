# backend/src/services/search_service.py
import logging
from typing import List, Optional, Dict, Any, Set
from sentence_transformers import SentenceTransformer, util
from pymilvus import utility
import numpy as np
from scipy.sparse import coo_matrix
from src.services.embeddings import sparse_embedding_model
from src.db.milvus_client import MilvusClient
from src.core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        # Connect to Milvus
        try:
            logger.debug("Connecting to Milvus for search service")
            self.client = MilvusClient()
            self.collection = self.client.get_collection()
        except Exception as e:
            logger.error("Failed to connect to Milvus: %s", e)
            self.collection = None

        # Load embedding models
        try:
            self.dense_encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error("Failed to load dense encoder: %s", e)
            self.dense_encoder = None

        self.sparse_encoder = sparse_embedding_model

        # Setup reranker
        if settings.ENABLE_RERANKER:
            try:
                self.reranker = SentenceTransformer(settings.RERANKER_MODEL_NAME)
            except Exception as e:
                logger.error("Failed to load reranker: %s", e)
                self.reranker = None
        else:
            self.reranker = None

    def _convert_sparse_to_milvus_format(self, sparse_matrix):
        """Convert sparse matrix to Milvus-compatible format."""
        if hasattr(sparse_matrix, 'tocoo'):
            coo = sparse_matrix.tocoo()
        else:
            coo = sparse_matrix
            
        # Handle different scipy versions
        if hasattr(coo, 'row') and hasattr(coo, 'col'):
            indices = coo.col.astype(np.int32)
            values = coo.data.astype(np.float32)
        elif hasattr(coo, 'indices'):
            indices = coo.indices.astype(np.int32)
            values = coo.data.astype(np.float32)
        else:
            dense = coo.toarray().flatten()
            indices = np.where(dense != 0)[0].astype(np.int32)
            values = dense[indices].astype(np.float32)
        
        return {"indices": indices.tolist(), "values": values.tolist()}

    def _extract_searchable_text(self, metadata: Dict[str, Any]) -> str:
        """
        Extract the most relevant text for reranking.
        Focus on product names, descriptions, not metadata noise.
        """
        # Priority order for text extraction
        text_fields = [
            "product_name", "name", "title", "description", 
            "part_number", "sku", "model", "item_name"
        ]
        
        searchable_parts = []
        
        # Add high-priority fields first
        for field in text_fields:
            if field in metadata and metadata[field]:
                searchable_parts.append(str(metadata[field]))
        
        # Add other relevant fields (but not IDs, URLs, weights, etc.)
        skip_fields = {
            "id", "url", "weight", "price", "timestamp", "created_at", 
            "updated_at", "row_index", "partition", "file_path"
        }
        
        for key, value in metadata.items():
            if (key not in text_fields and 
                key.lower() not in skip_fields and 
                value and 
                isinstance(value, (str, int, float)) and
                len(str(value)) > 2):  # Skip very short values
                searchable_parts.append(f"{key}: {value}")
        
        return " | ".join(searchable_parts)

    def _deduplicate_by_content(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate hits that represent the same logical item across partitions.
        Keep the one with the highest score.
        """
        # Group by content similarity (you might want to use row_index or a content hash)
        seen_content: Dict[str, Dict[str, Any]] = {}
        
        for hit in hits:
            # Create a content key - could be row_index, product_name, or content hash
            content_key = None
            
            # Try row_index first (most reliable)
            if "row_index" in hit:
                content_key = f"row_{hit['row_index']}"
            # Fall back to product name/title
            elif any(field in hit for field in ["product_name", "name", "title"]):
                for field in ["product_name", "name", "title"]:
                    if field in hit and hit[field]:
                        content_key = f"name_{hit[field]}"
                        break
            # Last resort: use the text content
            else:
                text_content = hit.get("text", "")[:100]  # First 100 chars
                content_key = f"text_{hash(text_content)}"
            
            if content_key:
                # Keep the hit with the highest score
                current_score = hit.get("score", 0)
                if (content_key not in seen_content or 
                    current_score > seen_content[content_key].get("score", 0)):
                    seen_content[content_key] = hit
        
        return list(seen_content.values())

    def search(
        self,
        query: str,
        top_k: int = 10,
        expr: Optional[str] = None,
        enable_bm25: bool = True,
        bm25_weight: float = 2.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with proper BM25, dense, and sparse search.
        """
        try:
            logger.debug("Starting search for '%s'", query)
            all_hits = []
            
            # 1) BM25 Search (if enabled and available)
            if enable_bm25:
                try:
                    bm25_results = self.collection.search(
                        data=[query],
                        anns_field="text",
                        param={
                            "metric_type": "BM25",
                            "params": {
                                "k1": 1.2,      # Term frequency saturation
                                "b": 0.75       # Length normalization
                            }
                        },
                        limit=top_k * 2,  # Get more candidates
                        expr=expr,
                        output_fields=["metadata", "text"]
                    )
                    
                    if bm25_results and len(bm25_results) > 0:
                        for hit in bm25_results[0]:
                            hit_data = {
                                **hit.entity.get("metadata", {}),
                                "text": hit.entity.get("text", ""),
                                "score": float(hit.score) * bm25_weight,
                                "search_type": "bm25"
                            }
                            all_hits.append(hit_data)
                    logger.debug("BM25 search returned %d hits", len(bm25_results[0]))

                except Exception as e:
                    logger.warning(f"BM25 search failed: {e}")
            
            # 2) Dense Vector Search
            try:
                dense_q = self.dense_encoder.encode(query).tolist()
                dense_results = self.collection.search(
                    data=[dense_q],
                    anns_field="dense",
                    param={
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    },
                    limit=top_k * 2,
                    expr=expr,
                    output_fields=["metadata", "text"]
                )
                
                if dense_results and len(dense_results) > 0:
                    for hit in dense_results[0]:
                        hit_data = {
                            **hit.entity.get("metadata", {}),
                            "text": hit.entity.get("text", ""),
                            "score": float(hit.score) * dense_weight,
                            "search_type": "dense"
                        }
                        all_hits.append(hit_data)
                    logger.debug("Dense search returned %d hits", len(dense_results[0]))

            except Exception as e:
                logger.warning(f"Dense search failed: {e}")
            
            # 3) Sparse Vector Search (SPLADE)
            try:
                sparse_raw = self.sparse_encoder.encode(query)
                sparse_q = self._convert_sparse_to_milvus_format(sparse_raw)
                
                sparse_results = self.collection.search(
                    data=[sparse_q],
                    anns_field="sparse",
                    param={
                        "metric_type": "IP",
                        "params": {"drop_ratio": 0.1}  # Lower drop_ratio for more exact matches
                    },
                    limit=top_k * 2,
                    expr=expr,
                    output_fields=["metadata", "text"]
                )
                
                if sparse_results and len(sparse_results) > 0:
                    for hit in sparse_results[0]:
                        hit_data = {
                            **hit.entity.get("metadata", {}),
                            "text": hit.entity.get("text", ""),
                            "score": float(hit.score) * sparse_weight,
                            "search_type": "sparse"
                        }
                        all_hits.append(hit_data)
                    logger.debug("Sparse search returned %d hits", len(sparse_results[0]))
                        
            except Exception as e:
                logger.warning(f"Sparse search failed: {e}")
            
            if not all_hits:
                return []
            
            # 4) Deduplicate across partitions BEFORE reranking
            unique_hits = self._deduplicate_by_content(all_hits)
            logger.debug("After deduplication: %d hits", len(unique_hits))
            
            # 5) Combine scores from different search methods
            # Group by content key and combine scores
            final_hits = self._combine_search_scores(unique_hits)
            logger.debug("After score combination: %d hits", len(final_hits))
            
            # 6) Rerank using clean, relevant text
            if self.reranker and final_hits:
                final_hits = self._rerank_with_clean_text(query, final_hits)
                logger.debug("After reranking")
            
            # 7) Final sort and limit
            final_hits.sort(
                key=lambda x: x.get("rerank_score", x.get("combined_score", x.get("score", 0))),
                reverse=True
            )
            logger.debug("Returning top %d results", top_k)
            return final_hits[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _combine_search_scores(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine scores from different search methods for the same item.
        """
        combined: Dict[str, Dict[str, Any]] = {}
        
        for hit in hits:
            # Use row_index or create a unique key
            key = hit.get("row_index", f"hit_{hash(str(hit))}")
            
            if key not in combined:
                combined[key] = hit.copy()
                combined[key]["combined_score"] = hit.get("score", 0)
                combined[key]["search_methods"] = [hit.get("search_type", "unknown")]
            else:
                # Combine scores (you can adjust this strategy)
                existing_score = combined[key].get("combined_score", 0)
                new_score = hit.get("score", 0)
                
                # Simple addition (you might want to use weighted combination)
                combined[key]["combined_score"] = existing_score + new_score
                combined[key]["search_methods"].append(hit.get("search_type", "unknown"))
                
                # Keep the highest individual score as backup
                if new_score > combined[key].get("score", 0):
                    combined[key]["score"] = new_score
        
        return list(combined.values())

    def _rerank_with_clean_text(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank using only the most relevant text fields, not full metadata dumps.
        """
        try:
            if not docs:
                return docs
            
            # Extract clean, searchable text for each document
            clean_texts = []
            for doc in docs:
                clean_text = self._extract_searchable_text(doc)
                clean_texts.append(clean_text)
            
            # Encode query and documents
            all_texts = [query] + clean_texts
            embeddings = self.reranker.encode(all_texts)
            
            query_emb = embeddings[0]
            doc_embs = embeddings[1:]
            
            # Calculate similarities
            similarities = util.cos_sim(query_emb, doc_embs)[0]
            
            # Add rerank scores
            reranked_docs = []
            for i, doc in enumerate(docs):
                scored_doc = doc.copy()
                scored_doc["rerank_score"] = float(similarities[i])
                scored_doc["clean_text_used"] = clean_texts[i]  # For debugging
                reranked_docs.append(scored_doc)
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return docs with combined_score as rerank_score
            for doc in docs:
                doc["rerank_score"] = doc.get("combined_score", doc.get("score", 0.0))
            return docs

    def debug_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Debug version that shows what each search method returns.
        """
        logger.debug("Running debug search for '%s'", query)
        debug_info = {
            "query": query,
            "bm25_results": [],
            "dense_results": [],
            "sparse_results": [],
            "combined_results": []
        }
        
        # Test BM25
        try:
            bm25_results = self.collection.search(
                data=[query],
                anns_field="text",
                param={
                    "metric_type": "BM25",
                    "params": {"k1": 1.2, "b": 0.75}
                },
                limit=top_k,
                output_fields=["metadata", "text"]
            )
            
            if bm25_results and len(bm25_results) > 0:
                for hit in bm25_results[0]:
                    debug_info["bm25_results"].append({
                        "score": float(hit.score),
                        "text": hit.entity.get("text", "")[:200],
                        "metadata": hit.entity.get("metadata", {})
                    })
        except Exception as e:
            debug_info["bm25_error"] = str(e)
        
        # Test regular search
        regular_results = self.search(query, top_k)
        debug_info["combined_results"] = regular_results

        return debug_info

    def health_check(self) -> Dict[str, Any]:
        """Check if all components are working properly."""
        logger.debug("Running health check")
        status = {
            "milvus_connected": False,
            "dense_encoder_loaded": False,
            "sparse_encoder_loaded": False,
            "reranker_loaded": False,
            "bm25_supported": False
        }
        
        try:
            status["milvus_connected"] = self.collection is not None
            status["dense_encoder_loaded"] = self.dense_encoder is not None
            status["sparse_encoder_loaded"] = self.sparse_encoder is not None
            status["reranker_loaded"] = self.reranker is not None
            
            # Test BM25 support
            try:
                test_results = self.collection.search(
                    data=["test"],
                    anns_field="text",
                    param={"metric_type": "BM25", "params": {"k1": 1.2, "b": 0.75}},
                    limit=1
                )
                status["bm25_supported"] = True
            except:
                status["bm25_supported"] = False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return status