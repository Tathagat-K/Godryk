# src/main.py

from fastapi import FastAPI
import src.core.logger  # configure root logger on import
import logging

from src.api.routes import router
from src.api.routes import router as health_router
from src.api.routes.ingest_router import router as ingest_router
from src.api.routes.search_router import router as search_router

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG FastAPI")

app.include_router(router)
app.include_router(ingest_router)
app.include_router(health_router)
app.include_router(search_router)

@app.get("/")
async def read_root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to the RAG FastAPI"}
