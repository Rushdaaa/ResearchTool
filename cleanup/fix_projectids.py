#!/usr/bin/env python3
"""
Fix the ProjectID issue and remove duplicates correctly from the full hcr.csv
"""

import pandas as pd

def fix_projectids():
    """Fix the ProjectID numbering issue and then remove duplicates."""
    
    # Read the file with proper handling
    with open("datasets/promise/hcr.csv", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    print(f"Data lines (excluding header): {len(lines) - 1}")
    
    # Try reading with pandas but handle potential issues
    try:
        df = pd.read_csv("datasets/promise/hcr.csv", encoding='utf-8')
        print(f"Pandas successfully read: {len(df)} rows")
        print(f"ProjectID range: {df['ProjectID'].min()} to {df['ProjectID'].max()}")
        
        # Check if we have the expected 600 rows
        if len(df) != 600:
            print(f"WARNING: Expected 600 rows but got {len(df)}")
            
        # Now remove duplicates at rows 101 and 102 (0-based index 100, 101)
        print("\nRemoving duplicates at rows 101 and 102...")
        
        # Remove rows at index 100 and 101 (ProjectID 101 and 102)
        if len(df) >= 102:  # Make sure we have enough rows
            df_clean = df.drop(df.index[[100, 101]]).reset_index(drop=True)
            
            # Renumber ProjectIDs from 1
            df_clean['ProjectID'] = range(1, len(df_clean) + 1)
            
            # Save the cleaned file
            df_clean.to_csv("datasets/promise/hcr_clean.csv", index=False)
            
            print(f"Original: {len(df)} requirements")
            print(f"After removing duplicates: {len(df_clean)} requirements")
            print(f"Removed 2 duplicate rows")
            print(f"ProjectIDs renumbered 1-{len(df_clean)}")
            print("Saved to: datasets/promise/hcr_clean.csv")
        else:
            print(f"ERROR: File doesn't have enough rows to remove duplicates at positions 101-102")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    fix_projectids()
