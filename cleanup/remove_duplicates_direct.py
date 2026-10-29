#!/usr/bin/env python3
"""
Direct removal of rows 101 and 102 (duplicates of rows 1 and 2) and renumber ProjectIDs.
"""

import pandas as pd

def remove_specific_duplicates(input_file, output_file):
    """Remove rows 101 and 102 and renumber ProjectIDs sequentially."""
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original: {len(df)} requirements")
    
    # Remove rows at index 100 and 101 (ProjectID 101 and 102)
    rows_to_remove = [100, 101]  # 0-based index for rows 101 and 102
    df_clean = df.drop(df.index[rows_to_remove]).reset_index(drop=True)
    
    # Renumber ProjectIDs sequentially from 1
    df_clean['ProjectID'] = range(1, len(df_clean) + 1)
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    
    print(f"Removed rows 101 and 102 (duplicates)")
    print(f"Final: {len(df_clean)} requirements")
    print(f"ProjectIDs renumbered 1-{len(df_clean)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    remove_specific_duplicates("datasets/promise/hcr.csv", "datasets/promise/hcr_clean.csv")
    print("✅ Done!")
