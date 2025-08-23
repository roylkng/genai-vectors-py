# S3 Vectors Test Notebooks

This directory contains two Jupyter notebooks for testing the S3 Vectors API:

## 1. Basic Functionality Test (`basic_functionality_test.ipynb`)

A simple demonstration of core S3 Vectors functionality with a small dataset:
- Setting up the S3 Vectors client using our custom implementation
- Creating embeddings with local embedding service (LM Studio/Ollama)
- Basic vector operations (insert/search)
- Semantic similarity search with 5 sample documents

**Features tested:**
- Bucket and index creation
- Document insertion with metadata
- Semantic search
- Metadata retrieval
- Basic API operations

## 2. Large-Scale Test (`large_scale_test.ipynb`)

Comprehensive testing of S3 Vectors with large datasets and IVFPQ indexing:
- Large-scale vector operations (10,000+ vectors)
- IVFPQ index creation and optimization
- Performance testing with real-world datasets
- Batch processing and efficient data handling
- Metadata filtering at scale

**Features tested:**
- Large-scale vector insertion (10k+ vectors)
- IVFPQ index creation for efficient search
- Performance benchmarking
- Metadata filtering with large datasets
- System health checks

## Prerequisites

Before running the notebooks, ensure you have:

1. **S3 Vectors API Server Running:**
   ```bash
   cd /home/rajan/Desktop/work/genai-vectors-py
   source .venv/bin/activate
   python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Local Embedding Service (Optional but Recommended):**
   - **LM Studio:** Download from https://lmstudio.ai/
   - **Ollama:** Install from https://ollama.com/
   
   If neither is available, the notebooks will fall back to random embeddings.

3. **Jupyter Notebook:**
   ```bash
   pip install notebook
   ```

## Running the Notebooks

1. **Start Jupyter:**
   ```bash
   cd /home/rajan/Desktop/work/genai-vectors-py
   source .venv/bin/activate
   jupyter notebook
   ```

2. **Open the notebooks in your browser:**
   - Navigate to `http://localhost:8888`
   - Open `basic_functionality_test.ipynb` for basic testing
   - Open `large_scale_test.ipynb` for large-scale testing

## Expected Results

### Basic Functionality Test
- Creates a bucket and index
- Inserts 5 sample documents with metadata
- Performs semantic search on test queries
- Shows ranked results by similarity score
- Demonstrates metadata filtering

### Large-Scale Test
- Generates and inserts 10,000+ vectors
- Creates IVFPQ index for efficient search
- Benchmarks search performance
- Tests metadata filtering at scale
- Shows system health metrics

## Troubleshooting

### Common Issues

1. **Connection Errors:**
   - Ensure the S3 Vectors API server is running on `http://localhost:8000`
   - Check that the server is accessible: `curl http://localhost:8000/health`

2. **Embedding Service Not Available:**
   - The notebooks will automatically fall back to random embeddings
   - For better results, install LM Studio or Ollama locally

3. **Permission Errors:**
   - Ensure you're running in the virtual environment: `source .venv/bin/activate`
   - Check file permissions in the project directory

### Health Checks

You can verify the system is working by running:

```bash
# Check API health
curl http://localhost:8000/health

# Check system health  
curl http://localhost:8000/healthz
```

Both should return JSON responses indicating the system is healthy.

## Performance Notes

### Basic Test
- Runs quickly (under 1 minute)
- Uses small dataset (5 documents)
- Good for verifying basic functionality

### Large-Scale Test
- Takes longer (5-15 minutes depending on hardware)
- Uses larger dataset (10,000+ vectors)
- Good for performance benchmarking
- Tests IVFPQ indexing capabilities

The large-scale test can be adjusted by modifying the `total_vectors` variable in the notebook.