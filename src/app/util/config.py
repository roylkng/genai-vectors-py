import os

# Lance Configuration - Smart indexing like LanceDB
LANCE_INDEX_TYPE = os.getenv("LANCE_INDEX_TYPE", "AUTO")  # AUTO, IVF_PQ, HNSW, or NONE
LANCE_INDEX_THRESHOLD = int(os.getenv("LANCE_INDEX_THRESHOLD", "50000"))  # Index after 50k vectors

# S3/MinIO Configuration
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin123")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET_PREFIX = os.getenv("S3_BUCKET_PREFIX", "vb-")

# Lance-specific S3 Configuration
LANCE_S3_ENDPOINT = os.getenv("LANCE_S3_ENDPOINT", S3_ENDPOINT_URL)
LANCE_S3_REGION = os.getenv("LANCE_S3_REGION", S3_REGION)
LANCE_ACCESS_KEY = os.getenv("LANCE_ACCESS_KEY", S3_ACCESS_KEY)
LANCE_SECRET_KEY = os.getenv("LANCE_SECRET_KEY", S3_SECRET_KEY)
LANCE_ALLOW_HTTP = os.getenv("LANCE_ALLOW_HTTP", "true").lower() == "true"

# API Limits
MAX_BATCH = int(os.getenv("MAX_BATCH", "500"))
MAX_TOPK = int(os.getenv("MAX_TOPK", "30"))
MAX_DIM = int(os.getenv("MAX_DIM", "4096"))
MAX_FILTERABLE_BYTES = int(os.getenv("MAX_FILTERABLE_BYTES", "2048"))
MAX_TOTAL_METADATA_BYTES = int(os.getenv("MAX_TOTAL_METADATA_BYTES", "40960"))  # 40 KB
MAX_SEGMENT_COUNT = int(os.getenv("MAX_SEGMENT_COUNT", "16"))

# Object Layout
INDEX_DIR = "indexes"
META_DIR = "_meta"
BUCKET_CONFIG_KEY = "bucket.json"
POLICY_CONFIG_KEY = "policy.json"
INDEX_CONFIG_KEY = "_index_config.json"
TABLE_DIR = "table"
