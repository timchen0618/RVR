import json
import random
from typing import Dict, List

def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data: List[Dict], file_path: str):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def concatenate_question_keywords(ground_truth_file: str, 
                                 keywords_file: str, 
                                 output_file: str,
                                 sample_size: int = 10000,
                                 seed: int = 42):
    """
    Randomly sample training examples and concatenate questions with keywords.
    
    Args:
        ground_truth_file: Path to ground truth JSONL file with question_text
        keywords_file: Path to JSONL file with question and keywords fields
        output_file: Path to save the concatenated results
        sample_size: Number of samples to randomly select (default: 10000)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    print(f"Reading ground truth file: {ground_truth_file}")
    ground_truth_data = read_jsonl(ground_truth_file)
    
    print(f"Reading keywords file: {keywords_file}")
    keywords_data = read_jsonl(keywords_file)
    
    # Verify that both files have the same number of entries
    if len(ground_truth_data) != len(keywords_data):
        raise ValueError(f"Mismatch in number of entries: {len(ground_truth_data)} vs {len(keywords_data)}")
    
    print(f"Total available examples: {len(ground_truth_data)}")
    
    # Check if we have enough data to sample
    if sample_size > len(ground_truth_data):
        print(f"Warning: Requested sample size ({sample_size}) is larger than available data ({len(ground_truth_data)})")
        sample_size = len(ground_truth_data)
    
    # Generate random indices for sampling
    all_indices = list(range(len(ground_truth_data)))
    sampled_indices = random.sample(all_indices, sample_size)
    sampled_indices.sort()  # Sort to maintain some order in the output
    
    print(f"Randomly sampled {len(sampled_indices)} unique examples")
    
    # Process the sampled entries
    concatenated_data = []
    
    for idx in sampled_indices:
        gt_entry = ground_truth_data[idx]
        kw_entry = keywords_data[idx]
        
        # Verify that the questions match (as a sanity check)
        gt_question = gt_entry.get('question_text', '')
        kw_question = kw_entry.get('question', '')
        
        if gt_question != kw_question:
            print(f"Warning: Question mismatch at index {idx}")
            print(f"  Ground truth: {gt_question[:50]}...")
            print(f"  Keywords file: {kw_question[:50]}...")
        
        # Create new entry with concatenated question_text
        new_entry = gt_entry.copy()  # Copy all fields from ground truth
        
        # Concatenate question with keywords
        keywords = kw_entry.get('keywords', '')
        if keywords:
            # Format: "Original question. Keywords: keyword1, keyword2, ..."
            concatenated_question = f"{gt_question} Keywords: {keywords}"
        else:
            # If no keywords, just use the original question
            concatenated_question = gt_question
        
        # Update the question_text field
        new_entry['question_text'] = concatenated_question
        
        concatenated_data.append(new_entry)
    
    # Save the results
    print(f"Saving {len(concatenated_data)} entries to: {output_file}")
    write_jsonl(concatenated_data, output_file)
    
    # Print some statistics
    print("\n=== Processing Statistics ===")
    print(f"Original dataset size: {len(ground_truth_data)}")
    print(f"Sampled entries: {len(concatenated_data)}")
    print(f"Sampling percentage: {len(concatenated_data)/len(ground_truth_data)*100:.2f}%")
    
    # Show a few examples
    print("\n=== Sample Results (first 3) ===")
    for i, entry in enumerate(concatenated_data[:3]):
        original_idx = sampled_indices[i]
        original_q = ground_truth_data[original_idx].get('question_text', '')
        new_q = entry.get('question_text', '')
        print(f"\nExample {i+1} (original index: {original_idx}):")
        print(f"  Original: {original_q[:100]}...")
        print(f"  With keywords: {new_q[:150]}...")

def main():
    # Configuration
    GROUND_TRUTH_FILE = "../train_data.jsonl"  # Your ground truth file
    KEYWORDS_FILE = "train_data_keywords.json"  # Your file with keywords
    OUTPUT_FILE = "../train_data_10k_query+keywords.jsonl"  # Output file
    SAMPLE_SIZE = 10000  # Number of samples to take
    RANDOM_SEED = 42  # Set seed for reproducibility
    
    try:
        concatenate_question_keywords(
            ground_truth_file=GROUND_TRUTH_FILE,
            keywords_file=KEYWORDS_FILE,
            output_file=OUTPUT_FILE,
            sample_size=SAMPLE_SIZE,
            seed=RANDOM_SEED
        )
        print(f"\n✓ Successfully created training file with {SAMPLE_SIZE} samples")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()