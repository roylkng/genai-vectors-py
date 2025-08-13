# s3vec-py — S3‑compatible vector store (FastAPI)

This is a reference implementation of an S3‑like vector store:
- S3‑compatible API surface for vector buckets & indexes
- Callback-based indexing (no cron)
- Hybrid ANN (HNSW for small, IVF‑PQ‑like for large; implemented with sklearn)
- Slice storage in Parquet (fallback to JSONL)
- Filterable vs non‑filterable metadata (simple in-process filter engine)

> Note: For portability this demo uses scikit‑learn to simulate IVF‑PQ and a simple HNSW placeholder.
> Replace `index/ivfpq_backend.py` with Faiss and `index/hnsw_backend.py` with hnswlib for production.

## Quickstart

```bash
# using uv
uv venv
uv pip install -e .[dev]
export S3_ENDPOINT_URL=http://localhost:9000
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export S3_BUCKET_PREFIX=vec-

uv run uvicorn app.main:app --reload
```

Run tests:
```bash
uv run pytest -q
```

Bring up MinIO with docker compose if you like:
```bash
docker compose up -d
```

