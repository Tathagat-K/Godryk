# RAG FastAPI

Basic folder structure for a Retrieval-Augmented Generation (RAG) project using FastAPI.

## Structure

```
src/
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
uvicorn src.main:app --reload
```

## Setup

1. Create a `.env` file in the project root. A template is provided in the repository; adjust the values for your environment.
2. Install dependencies using **Poetry**:

```bash
poetry install
```

   Or with **pip**:

```bash
pip install -r requirements.txt
```
If you do not have a `requirements.txt` yet, generate one with:

```bash
poetry export -f requirements.txt --output requirements.txt
```
