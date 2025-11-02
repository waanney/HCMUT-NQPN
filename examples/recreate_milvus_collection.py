"""Script to drop and recreate Milvus collection with correct dimension.

This script is useful when:
- You changed embedding model (e.g., from multilingual-e5-base to multilingual-e5-large)
- Collection has wrong dimension (e.g., 768 vs 1024)
- You need to reset the collection

WARNING: This will DELETE all existing data in the collection!
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymilvus import utility, connections, Collection
from core.config import load_config
from db.milvus_client import (
    DOC_COLLECTION_NAME,
    MILVUS_ALIAS,
    MILVUS_URI,
    MILVUS_DB_NAME,
    connect_to_milvus,
    _build_docs_schema,
)

config = load_config()
expected_dim = config.milvus.doc_dense_dim

print(f"⚠️  WARNING: This will DELETE all data in collection '{DOC_COLLECTION_NAME}'")
print(f"Expected dimension: {expected_dim}")
print(f"Collection name: {DOC_COLLECTION_NAME}")
response = input("Type 'YES' to continue: ")

if response != "YES":
    print("Cancelled.")
    sys.exit(0)

try:
    # Connect to Milvus
    alias = connect_to_milvus()
    print(f"Connected to Milvus: {alias}")
    
    # Check if collection exists
    if utility.has_collection(DOC_COLLECTION_NAME, using=alias):
        print(f"Collection '{DOC_COLLECTION_NAME}' exists. Checking dimension...")
        existing_collection = Collection(DOC_COLLECTION_NAME, using=alias)
        
        # Get existing dimension
        existing_dim = None
        for field in existing_collection.schema.fields:
            if field.name == "dense_vec":
                existing_dim = getattr(field, "dim", None)
                break
        
        if existing_dim:
            print(f"Existing dimension: {existing_dim}")
            print(f"Expected dimension: {expected_dim}")
            
            if existing_dim == expected_dim:
                print(f"✅ Collection already has correct dimension ({expected_dim}). No need to recreate.")
                sys.exit(0)
        
        # Drop collection
        print(f"\nDropping collection '{DOC_COLLECTION_NAME}'...")
        utility.drop_collection(DOC_COLLECTION_NAME, using=alias)
        print("✅ Collection dropped successfully.")
    else:
        print(f"Collection '{DOC_COLLECTION_NAME}' does not exist. Will create new one.")
    
    # Create new collection with correct dimension
    print(f"\nCreating new collection '{DOC_COLLECTION_NAME}' with dimension {expected_dim}...")
    schema = _build_docs_schema(expected_dim)
    new_collection = Collection(
        name=DOC_COLLECTION_NAME,
        schema=schema,
        using=alias,
        consistency_level="Session",
    )
    
    print(f"✅ Collection '{DOC_COLLECTION_NAME}' created successfully with dimension {expected_dim}")
    print(f"\nYou can now ingest data. The collection is ready for embeddings with dimension {expected_dim}.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

