#!/usr/bin/env python3
"""
Check for duplicates and redundancy in hcr.csv
"""

import csv
import pandas as pd

def check_duplicates():
    # Read the file
    df = pd.read_csv('datasets/promise/hcr.csv')
    
    print(f"=== HCR.CSV DUPLICATE ANALYSIS ===")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print()

    # Check for exact duplicates (all columns)
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")

    # Check for duplicate requirement texts
    req_duplicates = df['RequirementText'].duplicated().sum()
    print(f"Duplicate requirement texts: {req_duplicates}")

    if req_duplicates > 0:
        print("\n--- DUPLICATE REQUIREMENT TEXTS FOUND ---")
        duplicate_mask = df['RequirementText'].duplicated(keep=False)
        duplicate_texts = df[duplicate_mask]['RequirementText'].unique()
        
        for i, text in enumerate(duplicate_texts[:10]):  # Show first 10
            matching_rows = df[df['RequirementText'] == text]
            print(f"\n{i+1}. Text: \"{text[:80]}...\"")
            print(f"   Appears in rows: {list(matching_rows.index + 1)}")
            for _, row in matching_rows.iterrows():
                print(f"   Row {row.name + 1}: ProjectID={row['ProjectID']}")

    # Check for duplicate ProjectIDs
    pid_duplicates = df['ProjectID'].duplicated().sum()
    print(f"\nDuplicate ProjectIDs: {pid_duplicates}")

    if pid_duplicates > 0:
        print("\n--- DUPLICATE PROJECT IDS FOUND ---")
        pid_duplicate_mask = df['ProjectID'].duplicated(keep=False)
        duplicate_pids = df[pid_duplicate_mask]['ProjectID'].unique()
        
        for pid in duplicate_pids[:5]:  # Show first 5
            matching_rows = df[df['ProjectID'] == pid]
            print(f"\nProjectID {pid} appears {len(matching_rows)} times:")
            for _, row in matching_rows.iterrows():
                print(f"  Row {row.name + 1}: \"{row['RequirementText'][:60]}...\"")

    # Check for similar texts (first 40 characters)
    print(f"\n--- CHECKING FOR SIMILAR TEXTS ---")
    text_prefixes = df['RequirementText'].str[:40]
    similar_duplicates = text_prefixes.duplicated().sum()
    print(f"Requirements with similar beginnings (first 40 chars): {similar_duplicates}")

    if similar_duplicates > 0:
        print("\n--- SIMILAR REQUIREMENT BEGINNINGS ---")
        similar_mask = text_prefixes.duplicated(keep=False)
        duplicate_prefixes = text_prefixes[similar_mask].unique()
        
        for i, prefix in enumerate(duplicate_prefixes[:5]):  # Show first 5
            matching_indices = text_prefixes[text_prefixes == prefix].index
            print(f"\n{i+1}. Similar beginning: \"{prefix}...\"")
            print(f"   Found in rows: {list(matching_indices + 1)}")
            for idx in matching_indices:
                print(f"   Row {idx + 1}: \"{df.iloc[idx]['RequirementText'][:80]}...\"")

    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total requirements: {len(df)}")
    print(f"Unique requirement texts: {df['RequirementText'].nunique()}")
    print(f"Unique ProjectIDs: {df['ProjectID'].nunique()}")
    print(f"ProjectID range: {df['ProjectID'].min()} to {df['ProjectID'].max()}")
    
    # Check if ProjectIDs are sequential
    expected_pids = set(range(df['ProjectID'].min(), df['ProjectID'].max() + 1))
    actual_pids = set(df['ProjectID'])
    missing_pids = expected_pids - actual_pids
    if missing_pids:
        print(f"Missing ProjectIDs: {sorted(list(missing_pids))}")
    else:
        print("ProjectIDs appear to be sequential with no gaps")

if __name__ == "__main__":
    check_duplicates()
