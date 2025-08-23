#!/usr/bin/env python3
"""
Minimal test script to debug the bucket creation issue
"""

import sys
import os
sys.path.append('src')

from fastapi import FastAPI
from app.api import router

# Create minimal FastAPI app
app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal debug server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
