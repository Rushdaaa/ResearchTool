#!/usr/bin/env python3
"""
Verify the actual content of hcr.csv
"""

import pandas as pd

def verify_file():
    file_path = "datasets/promise/hcr.csv"
    
    # Count lines manually
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Manual line count: {len(lines)} total lines")
    print(f"Data lines (excluding header): {len(lines) - 1}")
    
    # Load with pandas
    df = pd.read_csv(file_path)
    print(f"Pandas row count: {len(df)} rows")
    print(f"ProjectID range: {df['ProjectID'].min()} to {df['ProjectID'].max()}")
    
    # Check for any missing ProjectIDs
    expected_ids = set(range(1, len(df) + 1))
    actual_ids = set(df['ProjectID'].tolist())
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids
    
    if missing_ids:
        print(f"Missing ProjectIDs: {sorted(missing_ids)}")
    if extra_ids:
        print(f"Extra ProjectIDs: {sorted(extra_ids)}")
    
    # Check last few rows
    print("\nLast 5 rows:")
    print(df.tail()[['ProjectID', 'RequirementText']].to_string())

if __name__ == "__main__":
    verify_file()
