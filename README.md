# RAG FastAPI

Basic folder structure for a Retrieval-Augmented Generation (RAG) project using FastAPI.

## Structure

```
app/
    main.py            # FastAPI application entrypoint
    api/
        routes/        # API routers
    services/          # Business logic (e.g., RAG service)
    models/            # Data models or database interfaces

data/                  # Place to store data sets or embeddings

docs/                  # Documentation

tests/                 # Unit tests
```

Run the API with:

```bash
uvicorn app.main:app --reload
```
