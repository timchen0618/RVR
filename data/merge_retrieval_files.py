#!/usr/bin/env python3
"""
Script to merge two retrieval JSON files.
Combines the first file (first 10k queries) with the second file (remaining queries).
"""

import json
import argparse
from pathlib import Path


def merge_jsonl_files(file1_path, file2_path, output_path):
    """
    Merge two JSONL files sequentially.
    
    Args:
        file1_path: Path to the first file (first 10k queries)
        file2_path: Path to the second file (newly retrieved queries)
        output_path: Path to save the merged output
    """
    print(f"Merging files:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    print(f"  Output: {output_path}")
    
    count_file1 = 0
    count_file2 = 0
    
    # Open output file for writing
    with open(output_path, 'w', encoding='utf-8') as outfile:
        
        # First, copy all lines from file1
        print(f"\nProcessing {file1_path}...")
        with open(file1_path, 'r', encoding='utf-8') as f1:
            for line_num, line in enumerate(f1, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Validate it's valid JSON
                    json_obj = json.loads(line)
                    # Write to output
                    outfile.write(line + '\n')
                    count_file1 += 1
                    
                    if count_file1 % 1000 == 0:
                        print(f"  Processed {count_file1} records from file 1")
                        
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping malformed JSON at line {line_num} in {file1_path}")
                    print(f"    Error: {e}")
                    continue
        
        print(f"  ✓ Completed file 1: {count_file1} records")
        
        # Then, append all lines from file2
        print(f"\nProcessing {file2_path}...")
        with open(file2_path, 'r', encoding='utf-8') as f2:
            for line_num, line in enumerate(f2, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Validate it's valid JSON
                    json_obj = json.loads(line)
                    # Write to output
                    outfile.write(line + '\n')
                    count_file2 += 1
                    
                    if count_file2 % 1000 == 0:
                        print(f"  Processed {count_file2} records from file 2")
                        
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping malformed JSON at line {line_num} in {file2_path}")
                    print(f"    Error: {e}")
                    continue
        
        print(f"  ✓ Completed file 2: {count_file2} records")
    
    # Summary
    total_records = count_file1 + count_file2
    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Records from file 1: {count_file1:,}")
    print(f"Records from file 2: {count_file2:,}")
    print(f"Total records:       {total_records:,}")
    print(f"Output file:         {output_path}")
    print(f"{'='*60}")


def verify_merge(output_path, expected_file1_count=None, expected_file2_count=None):
    """
    Verify the merged file by checking record count and validating JSON.
    
    Args:
        output_path: Path to the merged file
        expected_file1_count: Expected number of records from file 1
        expected_file2_count: Expected number of records from file 2
    """
    print(f"\nVerifying merged file: {output_path}")
    
    total_count = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                json.loads(line)
                total_count += 1
            except json.JSONDecodeError as e:
                print(f"  ERROR: Invalid JSON at line {line_num}")
                print(f"    Error: {e}")
                return False
    
    print(f"  ✓ All {total_count:,} records are valid JSON")
    
    if expected_file1_count and expected_file2_count:
        expected_total = expected_file1_count + expected_file2_count
        if total_count == expected_total:
            print(f"  ✓ Record count matches expected total: {expected_total:,}")
        else:
            print(f"  ⚠ WARNING: Record count mismatch!")
            print(f"    Expected: {expected_total:,}")
            print(f"    Actual:   {total_count:,}")
            print(f"    Difference: {total_count - expected_total:+,}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Merge two JSONL retrieval files sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge
  python merge_retrieval_files.py file1.json file2.json output.json
  
  # With verification
  python merge_retrieval_files.py file1.json file2.json output.json --verify
  
  # With expected counts
  python merge_retrieval_files.py file1.json file2.json output.json \\
      --verify --expected1 10000 --expected2 5000
        """
    )
    
    parser.add_argument('file1', type=str,
                        help='Path to first file (first 10k queries)')
    parser.add_argument('file2', type=str,
                        help='Path to second file (newly retrieved queries)')
    parser.add_argument('output', type=str,
                        help='Path to output merged file')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the merged file after creation')
    parser.add_argument('--expected1', type=int,
                        help='Expected number of records in file 1 (for verification)')
    parser.add_argument('--expected2', type=int,
                        help='Expected number of records in file 2 (for verification)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"Error: File not found: {file1_path}")
        return 1
    
    if not file2_path.exists():
        print(f"Error: File not found: {file2_path}")
        return 1
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge the files
    try:
        merge_jsonl_files(file1_path, file2_path, output_path)
    except Exception as e:
        print(f"\nError during merge: {e}")
        return 1
    
    # Verify if requested
    if args.verify:
        try:
            verify_merge(output_path, args.expected1, args.expected2)
        except Exception as e:
            print(f"\nError during verification: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())