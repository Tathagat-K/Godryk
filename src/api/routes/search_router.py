from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
import logging

from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])
service = SearchService()

@router.get("/")
async def search_endpoint(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100),
    expr: Optional[str] = Query(None, description="Milvus filter expression")
):
    """
    Perform a hybrid search with BM25, dense & sparse vectors, and optional reranking.
    """
    try:
        logger.info("Search requested: '%s' top_k=%s expr=%s", q, top_k, expr)
        results = service.search(q, top_k, expr)
        logger.info("Search returned %d results", len(results))
        return {"query": q, "results": results}
    except Exception as e:
        logger.error("Search failed for query '%s': %s", q, e)
        raise HTTPException(status_code=500, detail=str(e))
