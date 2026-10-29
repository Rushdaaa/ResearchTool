#!/usr/bin/env python3
"""
Remove exact duplicates from hcr.csv while maintaining sequential ProjectID numbering.
Only removes exact text duplicates, preserves similar requirements.
"""

import pandas as pd
import sys

def clean_exact_duplicates(input_file, output_file):
    """Remove exact duplicates and renumber ProjectIDs sequentially."""
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} requirements")
    
    # Find exact duplicates based on RequirementText (case-sensitive)
    duplicate_mask = df.duplicated(subset=['RequirementText'], keep='first')
    duplicates = df[duplicate_mask]
    
    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} exact duplicate(s):")
        for idx, row in duplicates.iterrows():
            original_idx = df[df['RequirementText'] == row['RequirementText']].index[0]
            print(f"  - Row {idx + 1} (ProjectID {row['ProjectID']}): duplicate of Row {original_idx + 1} (ProjectID {df.loc[original_idx, 'ProjectID']})")
            print(f"    Text: \"{row['RequirementText'][:80]}{'...' if len(row['RequirementText']) > 80 else ''}\"")
    else:
        print("\nNo exact duplicates found.")
        return
    
    # Remove exact duplicates (keep first occurrence)
    df_clean = df.drop_duplicates(subset=['RequirementText'], keep='first').copy()
    
    print(f"\nAfter removing exact duplicates: {len(df_clean)} requirements")
    print(f"Removed: {len(df) - len(df_clean)} duplicate requirements")
    
    # Renumber ProjectIDs sequentially from 1
    df_clean['ProjectID'] = range(1, len(df_clean) + 1)
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    
    print(f"\nCleaned dataset saved to: {output_file}")
    print(f"ProjectIDs renumbered from 1 to {len(df_clean)}")
    
    # Verify no duplicates remain
    remaining_duplicates = df_clean.duplicated(subset=['RequirementText']).sum()
    print(f"Verification: {remaining_duplicates} duplicate requirements remaining")
    
    return df_clean

if __name__ == "__main__":
    input_file = "datasets/promise/hcr.csv"
    output_file = "datasets/promise/hcr_deduplicated.csv"
    
    try:
        cleaned_df = clean_exact_duplicates(input_file, output_file)
        print("\n✅ Exact duplicate removal completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
