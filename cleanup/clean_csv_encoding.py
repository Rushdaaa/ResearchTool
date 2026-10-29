#!/usr/bin/env python3
"""
Clean CSV text by fixing character encoding issues
Specifically fixes \92 (escaped apostrophes) and other common encoding problems
"""

import csv
import re

def clean_text(text):
    """Clean text by fixing common encoding issues and formatting problems"""
    # Fix \92 (octal for apostrophe)
    text = text.replace('\\92', "'")
    
    # Fix other common encoding issues
    text = text.replace('\\91', "'")  # Left single quote
    text = text.replace('\\93', '"')  # Left double quote
    text = text.replace('\\94', '"')  # Right double quote
    text = text.replace('\\95', '—')  # Em dash/bullet
    text = text.replace('\\96', '–')  # En dash
    text = text.replace('\\97', '—')  # Em dash
    
    # Fix leading whitespace issues - remove excessive leading spaces
    text = text.lstrip(' ')
    
    return text

def clean_csv_file(input_file, output_file):
    """Clean a CSV file by fixing encoding issues"""
    
    cleaned_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        
        for row in reader:
            cleaned_row = [clean_text(cell) for cell in row]
            cleaned_rows.append(cleaned_row)
    
    # Write cleaned data
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    
    print(f"Cleaned {len(cleaned_rows)} rows")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    return len(cleaned_rows)

def main():
    # Clean both Promise datasets
    
    # Clean Promise+.csv
    print("=== Cleaning Promise+.csv ===")
    clean_csv_file(
        'datasets/promise/Promise+.csv',
        'datasets/promise/Promise+_cleaned.csv'
    )
    
    print("\n=== Cleaning PROMISE_exp.csv ===")
    clean_csv_file(
        'datasets/promise/PROMISE_exp.csv',
        'datasets/promise/PROMISE_exp_cleaned.csv'
    )
    
    print("\n=== Sample of cleaned data ===")
    with open('datasets/promise/Promise+_cleaned.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 132:  # Show the specific line you highlighted
                print(f"Row {i}: {row}")
                break

if __name__ == "__main__":
    main()
