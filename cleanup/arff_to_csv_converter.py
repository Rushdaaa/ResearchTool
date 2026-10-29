#!/usr/bin/env python3
"""
Clean ARFF to CSV converter for PROMISE dataset
Converts PROMISE_exp.arff to a clean CSV format
"""

import re
import csv
import os

def parse_arff_to_csv(arff_file_path, csv_file_path):
    """
    Convert ARFF file to CSV format
    
    Args:
        arff_file_path (str): Path to input ARFF file
        csv_file_path (str): Path to output CSV file
    """
    
    # Read the ARFF file
    with open(arff_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract attribute information
    attributes = []
    attribute_pattern = r'@ATTRIBUTE\s+(\w+)\s+(.+)'
    
    for match in re.finditer(attribute_pattern, content):
        attr_name = match.group(1)
        attr_type = match.group(2)
        attributes.append(attr_name)
        print(f"Found attribute: {attr_name} - {attr_type}")
    
    # Find the data section
    data_section = content.split('@DATA')[1].strip()
    
    # Parse data rows
    data_rows = []
    for line in data_section.split('\n'):
        line = line.strip()
        if line and not line.startswith('%'):  # Skip empty lines and comments
            # Handle two different formats:
            # Format 1: 1,'text',CLASS (original)
            # Format 2: 1,'text'CLASS (Promise+ format - missing comma)
            
            # Check if this looks like the Promise+ format (no comma before class)
            if "'" in line and line.count(',') == 1:
                # Promise+ format: ProjectID,'RequirementText'CLASS
                # Find the closing quote and extract class after it
                parts = []
                
                # Extract ProjectID (before first comma)
                first_comma = line.find(',')
                if first_comma > 0:
                    project_id = line[:first_comma].strip()
                    remainder = line[first_comma + 1:].strip()
                    
                    # Find the quoted text
                    if remainder.startswith("'"):
                        # Find the closing quote
                        closing_quote = remainder.rfind("'")
                        if closing_quote > 0:
                            requirement_text = remainder[1:closing_quote]
                            class_label = remainder[closing_quote + 1:].strip()
                            
                            parts = [project_id, requirement_text, class_label]
                
                if len(parts) == len(attributes):
                    data_rows.append(parts)
                    continue
            
            # Original format parsing with proper comma separation
            parts = []
            current_part = ""
            in_quotes = False
            
            i = 0
            while i < len(line):
                char = line[i]
                
                if char == "'" and (i == 0 or line[i-1] == ','):
                    # Start of quoted string
                    in_quotes = True
                    current_part = ""
                elif char == "'" and in_quotes:
                    # End of quoted string
                    in_quotes = False
                    parts.append(current_part)
                    current_part = ""
                    # Skip the comma if it follows
                    if i + 1 < len(line) and line[i + 1] == ',':
                        i += 1
                elif char == ',' and not in_quotes:
                    # Field separator outside quotes
                    if current_part:
                        parts.append(current_part)
                        current_part = ""
                else:
                    current_part += char
                
                i += 1
            
            # Add the last part
            if current_part:
                parts.append(current_part)
            
            if len(parts) == len(attributes):
                data_rows.append(parts)
            else:
                print(f"Warning: Skipping malformed line: {line[:50]}...")
    
    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(attributes)
        
        # Write data rows
        for row in data_rows:
            writer.writerow(row)
    
    print(f"\nConversion completed!")
    print(f"Input file: {arff_file_path}")
    print(f"Output file: {csv_file_path}")
    print(f"Attributes: {attributes}")
    print(f"Total rows converted: {len(data_rows)}")
    
    return len(data_rows)

def main():
    import sys
    
    # Allow command line argument for file selection
    if len(sys.argv) > 1:
        if sys.argv[1] == "plus":
            arff_file = r"c:\Users\khanr\Downloads\Desktop\ResearchTool\datasets\promise\Promise+.arff"
            csv_file = r"c:\Users\khanr\Downloads\Desktop\ResearchTool\datasets\promise\Promise+.csv"
        else:
            arff_file = sys.argv[1]
            csv_file = arff_file.replace('.arff', '.csv')
    else:
        # Default to PROMISE_exp.arff
        arff_file = r"c:\Users\khanr\Downloads\Desktop\ResearchTool\datasets\promise\PROMISE_exp.arff"
        csv_file = r"c:\Users\khanr\Downloads\Desktop\ResearchTool\datasets\promise\PROMISE_exp.csv"
    
    # Check if input file exists
    if not os.path.exists(arff_file):
        print(f"Error: Input file not found: {arff_file}")
        return
    
    # Perform conversion
    try:
        rows_converted = parse_arff_to_csv(arff_file, csv_file)
        
        # Display sample of converted data
        print(f"\nSample of converted data:")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 5:  # Show first 5 rows
                    print(f"Row {i}: {row}")
                else:
                    break
                    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main()
