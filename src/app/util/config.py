import os

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET_PREFIX = os.getenv("S3_BUCKET_PREFIX", "vec-")

MAX_BATCH = int(os.getenv("MAX_BATCH", "500"))          # vectors per PutVectors
MAX_TOPK = int(os.getenv("MAX_TOPK", "30"))             # topK per QueryVectors
MAX_DIM = int(os.getenv("MAX_DIM", "4096"))             # max dimension
MAX_FILTERABLE_BYTES = int(os.getenv("MAX_FILTERABLE_BYTES", "2048"))
MAX_TOTAL_METADATA_BYTES = int(os.getenv("MAX_TOTAL_METADATA_BYTES", "40960"))  # 40 KB

SLICE_ROW_LIMIT = int(os.getenv("SLICE_ROW_LIMIT", "50000"))
SLICE_AGE_LIMIT_S = int(os.getenv("SLICE_AGE_LIMIT_S", "30"))
SLICE_FORMAT = os.getenv("SLICE_FORMAT", "parquet")  # "parquet" or "jsonl"

# Index layout
INDEX_DIR = "indexes"
IDMAP_KEY = "idmap/idmap.parquet"
MANIFEST_KEY = "manifest.json"
STAGED_DIR = "staged"
