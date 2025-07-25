# src/main.py

from fastapi import FastAPI
from src.api.routes import router
from src.api.routes.ingest_router import router as ingest_router

app = FastAPI(title="RAG FastAPI")

app.include_router(router)
app.include_router(ingest_router)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG FastAPI"}
