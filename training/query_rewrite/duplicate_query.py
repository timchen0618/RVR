import json
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple

# Fix CUDA multiprocessing issue
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

class DualQueryGenerator:
    def __init__(self, model_name, max_context_length):
        print(f"Loading vLLM model for dual query generation: {model_name}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=4, 
            gpu_memory_utilization=0.9, 
            max_model_len=max_context_length,
            trust_remote_code=True,
            dtype="auto"
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,  
            max_tokens=100,   # Increased for two queries
            top_p=0.8,
            top_k=20,
        )
        
        self.max_context_length = max_context_length
        
        print("vLLM model loaded successfully for dual query generation!", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"Number of GPUs: {torch.cuda.device_count()}", flush=True)
    
    def create_dual_query_prompt(self, query: str) -> str:
        """Create prompt for generating two alternative queries"""
        
        prompt = f"""Given the following query, generate two alternative queries that ask for the same information but in different ways. 

Original Query: {query}

Instructions:
1. Maintain the same intent and seek the same answer
2. Use different wording, structure, or perspective
3. Be clear and natural-sounding questions
4. Generate exactly two alternative queries, one per line, without numbering or labels.

Alternative Query 1:
Alternative Query 2:"""
        
        return prompt

    def generate_dual_queries_batch(self, entries: List[Dict[str, Any]], 
                                   batch_size: int = 8) -> Tuple[List[str], List[str]]:
        """Generate two alternative queries for each input query in batches"""
        queries_1 = []
        queries_2 = []
        
        # Prepare all prompts
        all_prompts = []
        valid_indices = []
        
        for i, entry in enumerate(entries):
            query = entry.get("question", "") or entry.get("input", "") or entry.get("original_query", "")
            
            # Check if we have a valid query
            if not query:
                print(f"Warning: Entry {i+1} missing query", flush=True)
                queries_1.append("")
                queries_2.append("")
                continue
            
            prompt = self.create_dual_query_prompt(query)
            all_prompts.append(prompt)
            valid_indices.append(i)
        
        # Process in batches
        for i in range(0, len(all_prompts), batch_size):
            batch = all_prompts[i:i + batch_size]
            batch_indices = valid_indices[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}: queries {i+1}-{min(i+batch_size, len(all_prompts))}", flush=True)
            
            try:
                # Generate dual queries for the batch
                outputs = self.llm.generate(batch, self.sampling_params)
                
                # Process responses
                for output, orig_idx in zip(outputs, batch_indices):
                    generated_text = output.outputs[0].text.strip()
                    
                    # Parse the two queries from the response
                    lines = generated_text.split('\n')
                    
                    # Clean up and extract queries
                    query1 = ""
                    query2 = ""
                    
                    for line in lines:
                        line = line.strip()
                        # Skip empty lines and labels
                        if not line or line.startswith("Alternative Query"):
                            continue
                        
                        # Remove any numbering or labels
                        line = line.lstrip('1234567890.-: ')
                        
                        if not query1:
                            query1 = line
                        elif not query2:
                            query2 = line
                            break  # We have both queries
                    
                    # If we couldn't parse two queries, use the original
                    original = entries[orig_idx].get("question", "") or entries[orig_idx].get("input", "")
                    
                    if not query1:
                        query1 = original
                    if not query2:
                        query2 = original
                    
                    # Store results at correct indices
                    while len(queries_1) <= orig_idx:
                        queries_1.append("")
                    while len(queries_2) <= orig_idx:
                        queries_2.append("")
                    
                    queries_1[orig_idx] = query1
                    queries_2[orig_idx] = query2
                    
                    print(f"  ✓ Generated queries for entry {orig_idx+1}", flush=True)
                    print(f"    Original: {original[:50]}...", flush=True)
                    print(f"    Query 1: {query1[:50]}...", flush=True)
                    print(f"    Query 2: {query2[:50]}...", flush=True)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}", flush=True)
                # Keep original queries for failed batch
                for idx in batch_indices:
                    original = entries[idx].get("question", "") or entries[idx].get("input", "")
                    while len(queries_1) <= idx:
                        queries_1.append("")
                    while len(queries_2) <= idx:
                        queries_2.append("")
                    queries_1[idx] = original
                    queries_2[idx] = original
        
        # Fill in any missing entries with original queries
        for i, entry in enumerate(entries):
            original = entry.get("question", "") or entry.get("input", "")
            if i >= len(queries_1):
                queries_1.append(original)
            elif not queries_1[i]:
                queries_1[i] = original
            
            if i >= len(queries_2):
                queries_2.append(original)
            elif not queries_2[i]:
                queries_2[i] = original
        
        return queries_1, queries_2

def process_and_generate_dual_files(generator: DualQueryGenerator, 
                                   input_file: str, 
                                   output_file1: str,
                                   output_file2: str,
                                   batch_size: int = 8):
    """Process all JSON entries and create two output files with different queries"""
    
    # Load all entries
    all_entries = []
    print(f"Loading entries from {input_file}...", flush=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                all_entries.append(data)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}, skipping", flush=True)
                continue
    
    print(f"Loaded {len(all_entries)} entries", flush=True)
    
    # Generate dual queries for all entries
    queries_1, queries_2 = generator.generate_dual_queries_batch(
        all_entries, batch_size
    )
    
    # Create two sets of entries with different queries
    entries_set1 = []
    entries_set2 = []
    
    for i, (entry, q1, q2) in enumerate(zip(all_entries, queries_1, queries_2)):
        # Get original query for reference
        original_query = entry.get("question", "") or entry.get("input", "")
        
        # Create entry for file 1
        entry1 = {
            "question": q1,  # New query 1
            "original_query": original_query,  # Original query saved here
            "input": entry.get("input", ""),  # Keep input field if it exists
            "id": entry.get("id", i),  # Keep id field
            "query_version": 1  # Mark this as query version 1
        }
        
        # Create entry for file 2
        entry2 = {
            "question": q2,  # New query 2
            "original_query": original_query,  # Original query saved here
            "input": entry.get("input", ""),  # Keep input field if it exists
            "id": entry.get("id", i),  # Keep id field
            "query_version": 2  # Mark this as query version 2
        }
        
        # Add optional fields if they exist
        if "input_negative_ctxs" in entry:
            entry1["input_negative_ctxs"] = entry["input_negative_ctxs"]
            entry2["input_negative_ctxs"] = entry["input_negative_ctxs"]
        
        entries_set1.append(entry1)
        entries_set2.append(entry2)
    
    # Save to two separate files
    with open(output_file1, 'w', encoding='utf-8') as f:
        for entry in entries_set1:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    with open(output_file2, 'w', encoding='utf-8') as f:
        for entry in entries_set2:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n=== Generation Statistics ===")
    print(f"Total entries processed: {len(all_entries)}")
    print(f"Entries in file 1: {len(entries_set1)}")
    print(f"Entries in file 2: {len(entries_set2)}")
    
    # Calculate average query lengths
    if entries_set1:
        original_lengths = [len(e.get("original_query", "")) for e in entries_set1]
        q1_lengths = [len(e.get("question", "")) for e in entries_set1]
        q2_lengths = [len(e.get("question", "")) for e in entries_set2]
        
        avg_original = sum(original_lengths) / len(original_lengths) if original_lengths else 0
        avg_q1 = sum(q1_lengths) / len(q1_lengths) if q1_lengths else 0
        avg_q2 = sum(q2_lengths) / len(q2_lengths) if q2_lengths else 0
        
        print(f"\nQuery length statistics:")
        print(f"  Average original query length: {avg_original:.1f} characters")
        print(f"  Average query 1 length: {avg_q1:.1f} characters")
        print(f"  Average query 2 length: {avg_q2:.1f} characters")
    
    return entries_set1, entries_set2

def main():
    # Configuration
    INPUT_FILE = "contriever_base_finetuned_t200_MAX12.json"  # Your input file with original queries
    OUTPUT_FILE1 = "contriever_base_alternate_query_q1.json"  # Output file with first set of queries
    OUTPUT_FILE2 = "contriever_base_alternate_query_q2.json"  # Output file with second set of queries
    MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # Or your preferred model
    
    # Generation configuration
    BATCH_SIZE = 16  
    MAX_CONTEXT_LENGTH = 8096
    
    print("=== Dual Query Generation with vLLM ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file 1: {OUTPUT_FILE1}")
    print(f"Output file 2: {OUTPUT_FILE2}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Initialize generator
    generator = DualQueryGenerator(MODEL_NAME, MAX_CONTEXT_LENGTH)
    
    try:
        # Process entries and generate dual files
        entries_set1, entries_set2 = process_and_generate_dual_files(
            generator, 
            INPUT_FILE, 
            OUTPUT_FILE1,
            OUTPUT_FILE2,
            BATCH_SIZE
        )
        
        print(f"\n✓ Successfully generated dual queries for {len(entries_set1)} entries")
        print(f"✓ Results saved to:")
        print(f"  - {OUTPUT_FILE1} (Query version 1)")
        print(f"  - {OUTPUT_FILE2} (Query version 2)")
        
        # Show sample results
        if entries_set1 and entries_set2:
            print("\n=== Sample Results ===")
            for i in range(min(3, len(entries_set1))):
                print(f"\nExample {i+1}:")
                print(f"Original: {entries_set1[i].get('original_query', '')[:100]}...")
                print(f"Query 1: {entries_set1[i].get('question', '')[:100]}...")
                print(f"Query 2: {entries_set2[i].get('question', '')[:100]}...")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()