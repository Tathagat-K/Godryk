from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from src.services.search_service import SearchService

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
        results = service.search(q, top_k, expr)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
