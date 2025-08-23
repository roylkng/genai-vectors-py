#!/usr/bin/env python3
"""
Direct database test to see what's actually stored
"""

import sys
import os
sys.path.append('/home/rajan/Desktop/work/genai-vectors-py/src')

import lancedb
import pyarrow as pa
import json
import numpy as np
from app.lance.db import connect_bucket, table_path
from app.lance.schema import create_vector_schema, prepare_batch_data
from app.util import config

def test_direct_db():
    """Test the database directly to see what's being stored"""
    # Connect to Lance
    db = connect_bucket("debug-metadata-1755963439")
    table_uri = table_path("debug-index")
    
    print(f"Table URI: {table_uri}")
    
    try:
        table = db.open_table(table_uri)
        print(f"Table schema: {table.schema}")
        
        # Check the actual data
        df = table.to_pandas()
        print(f"Dataframe shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("First row:")
        for col in df.columns:
            print(f"  {col}: {df.iloc[0][col]}")
            
        # Check if metadata_json column exists and has data
        if 'metadata_json' in df.columns:
            print(f"metadata_json column exists")
            print(f"metadata_json value: {df.iloc[0]['metadata_json']}")
            if df.iloc[0]['metadata_json']:
                try:
                    metadata = json.loads(df.iloc[0]['metadata_json'])
                    print(f"Parsed metadata: {metadata}")
                except Exception as e:
                    print(f"Error parsing metadata: {e}")
        else:
            print("metadata_json column does not exist")
            
    except Exception as e:
        print(f"Error opening table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_db()