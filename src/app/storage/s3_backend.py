import io, json, time
from typing import Optional, Iterator, List, Dict, Any, Tuple
import boto3
from botocore.config import Config
from .slices import rows_to_parquet_bytes, rows_to_jsonl_bytes
from ..util import config

class S3Storage:
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT_URL,
            aws_access_key_id=config.S3_ACCESS_KEY,
            aws_secret_access_key=config.S3_SECRET_KEY,
            region_name=config.S3_REGION,
            config=Config(s3={"addressing_style": "path"}),
        )

    # ----- bucket helpers -----
    def bucket_name(self, vector_bucket: str) -> str:
        return f"{config.S3_BUCKET_PREFIX}{vector_bucket}"

    def ensure_bucket(self, vector_bucket: str) -> None:
        bn = self.bucket_name(vector_bucket)
        existing = [b["Name"] for b in self.client.list_buckets().get("Buckets", [])]
        if bn not in existing:
            self.client.create_bucket(Bucket=bn)

    def list_vector_buckets(self) -> List[str]:
        prefix = config.S3_BUCKET_PREFIX
        return [b["Name"][len(prefix):]
                for b in self.client.list_buckets().get("Buckets", [])
                if b["Name"].startswith(prefix)]

    # ----- generic object ops -----
    def put_json(self, vector_bucket: str, key: str, data: dict) -> None:
        bn = self.bucket_name(vector_bucket)
        self.client.put_object(Bucket=bn, Key=key,
                               Body=json.dumps(data).encode("utf-8"),
                               ContentType="application/json")

    def get_json(self, vector_bucket: str, key: str) -> Optional[dict]:
        bn = self.bucket_name(vector_bucket)
        try:
            obj = self.client.get_object(Bucket=bn, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None
        return json.loads(obj["Body"].read())

    def upload_bytes(self, vector_bucket: str, key: str, body: bytes, content_type: str="application/octet-stream") -> None:
        bn = self.bucket_name(vector_bucket)
        self.client.put_object(Bucket=bn, Key=key, Body=body, ContentType=content_type)

    def download_bytes(self, vector_bucket: str, key: str) -> bytes:
        bn = self.bucket_name(vector_bucket)
        obj = self.client.get_object(Bucket=bn, Key=key)
        return obj["Body"].read()

    def list_prefix(self, vector_bucket: str, prefix: str) -> Iterator[str]:
        bn = self.bucket_name(vector_bucket)
        cont = None
        while True:
            kw = {"Bucket": bn, "Prefix": prefix}
            if cont: kw["ContinuationToken"] = cont
            resp = self.client.list_objects_v2(**kw)
            for it in resp.get("Contents", []):
                yield it["Key"]
            if not resp.get("IsTruncated"): break
            cont = resp.get("NextContinuationToken")

    def delete_prefix(self, vector_bucket: str, prefix: str) -> None:
        bn = self.bucket_name(vector_bucket)
        keys = [{"Key": k} for k in self.list_prefix(vector_bucket, prefix)]
        if not keys: return
        # Delete in batches of 1000
        for i in range(0, len(keys), 1000):
            self.client.delete_objects(Bucket=bn, Delete={"Objects": keys[i:i+1000]})

    def delete_object(self, vector_bucket: str, key: str) -> None:
        bn = self.bucket_name(vector_bucket)
        try:
            self.client.delete_object(Bucket=bn, Key=key)
        except self.client.exceptions.NoSuchKey:
            pass  # Ignore if object doesn't exist

    # ----- layout helpers -----
    def index_config_key(self, index: str) -> str:
        return f"{config.INDEX_DIR}/{index}/config.json"

    def idmap_key(self, index: str) -> str:
        return f"{config.INDEX_DIR}/{index}/{config.IDMAP_KEY}"

    def manifest_key(self, index: str) -> str:
        return f"{config.INDEX_DIR}/{index}/{config.MANIFEST_KEY}"

    def staged_path(self, index: str, ext: str) -> str:
        ts = int(time.time()*1000)
        return f"{config.STAGED_DIR}/{index}/slice-{ts}.{ext}"

    def index_file_key(self, index: str, algo_ext: str) -> str:
        # IVF-PQ: "index.faiss"; HNSW: "index.hnsw"
        return f"{config.INDEX_DIR}/{index}/index.{algo_ext}"

    # ----- slices -----
    def write_slice(self, vector_bucket: str, index: str, rows: List[Dict[str, Any]]) -> str:
        ext = "parquet" if config.SLICE_FORMAT.lower() == "parquet" else "jsonl"
        key = self.staged_path(index, ext)
        if ext == "parquet":
            buf = rows_to_parquet_bytes(rows)
            self.upload_bytes(vector_bucket, key, buf.getvalue())
        else:
            buf = rows_to_jsonl_bytes(rows)
            self.upload_bytes(vector_bucket, key, buf.getvalue(), content_type="application/x-ndjson")
        return key
