# backend/src/api/ingest_router.py

from fastapi import APIRouter, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from pymilvus import utility
from src.services.ingestion_service import ingest_file
from src.db.milvus_client import MilvusClient

router = APIRouter(prefix="/ingest", tags=["ingestion"])

@router.get("/collections")
async def list_collections():
    """
    Ensure a connection exists, then list all Milvus collections.
    """
    try:
        # Establish connection (no-op if already connected)
        MilvusClient()

        cols = utility.list_collections()
        return {"collections": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file")
async def ingest_file_endpoint(
    upload: UploadFile,
    x_user_id: str = Header(..., description="ID of the user uploading the file"),
):
    """
    Ingest an uploaded file into Milvus.
    """
    try:
        # Make sure connection is alive
        MilvusClient()

        await ingest_file(upload, x_user_id)
        return JSONResponse(status_code=202, content={"detail": "Ingestion started"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
