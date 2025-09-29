import json
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional, Tuple

# Fix CUDA multiprocessing issue
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

class QueryRewriterVLLM:
    def __init__(self, model_name, max_context_length):
        print(f"Loading vLLM model for query rewriting: {model_name}")
        
        # Initialize vLLM with optimized settings
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Force single GPU
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=max_context_length,
            trust_remote_code=True,
            dtype="auto"
        )
        
        # Set up sampling parameters for query rewriting
        # More creative than verification task
        self.sampling_params = SamplingParams(
            temperature=0.7,  # Some creativity for rewriting
            max_tokens=64,   # Longer output for rewritten queries
            top_p=0.8,
            top_k=20,
        )
        
        self.max_context_length = max_context_length
        
        print("vLLM model loaded successfully for query rewriting!", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"Number of GPUs: {torch.cuda.device_count()}", flush=True)
    
    def create_rewrite_prompt(self, query: str, documents: List[Dict[str, Any]], 
                             rewrite_type: str = "comprehensive") -> str:
        """Create prompt for query rewriting based on gold documents"""
        
        # Prepare document context
        doc_contexts = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 documents for context
            title = doc.get('title', 'No title')
            text = doc.get('text', '')[:500]  # Limit text length
            score = doc.get('score', 'N/A')
            doc_contexts.append(f"Document {i} (Score: {score}):\nTitle: {title}\nText: {text}")
        
        documents_text = "\n\n".join(doc_contexts)
        
        # Different prompt templates based on rewrite type
        if rewrite_type == "comprehensive":
            prompt = f"""You are an expert query rewriter. Your task is to rewrite the given query to be more comprehensive and specific based on the retrieved documents.

Original Query: {query}

Retrieved Documents:
{documents_text}

Instructions:
1. Analyze the retrieved documents to understand the context and key information
2. Rewrite the query to be more specific and comprehensive
3. Include relevant entities, concepts, and context from the documents
4. Make the query more likely to retrieve relevant information
5. Keep the rewritten query clear and focused
6. DO NOT answer the question, only rewrite it

Rewritten Query:"""

        elif rewrite_type == "expansion":
            prompt = f"""You are an expert query expansion specialist. Expand the given query with relevant terms and concepts from the retrieved documents.

Original Query: {query}

Retrieved Documents:
{documents_text}

Instructions:
1. Identify key terms, entities, and concepts from the documents
2. Expand the original query by adding relevant context
3. Include synonyms and related terms that appear in the documents
4. Keep the expanded query focused on the original intent
5. DO NOT answer the question, only expand it

Expanded Query:"""

        else:  # "fusion" type - combine multiple rewriting strategies
            prompt = f"""You are an expert query rewriter. Rewrite the query using information from retrieved documents to make it more effective for information retrieval.

Original Query: {query}

Retrieved Documents:
{documents_text}

Instructions:
1. Analyze the documents to understand the context
2. Rewrite the query incorporating:
   - Key entities and concepts from the documents
   - More specific terminology
   - Relevant context that clarifies intent
3. Make the query more likely to retrieve comprehensive, relevant information
4. Keep it concise but informative
5. DO NOT answer the question, only rewrite it

Rewritten Query:"""

        return prompt

    def rewrite_queries_batch(self, entries: List[Dict[str, Any]], 
                            rewrite_type: str = "comprehensive", 
                            batch_size: int = 8) -> List[str]:
        """Rewrite multiple queries in batches using vLLM"""
        rewritten_queries = []
        
        # Prepare all prompts
        all_prompts = []
        valid_indices = []
        
        for i, entry in enumerate(entries):
            query = entry.get("question", "")
            documents = entry.get("ctxs", [])
            
            if query and documents:
                prompt = self.create_rewrite_prompt(query, documents, rewrite_type)
                all_prompts.append(prompt)
                valid_indices.append(i)
            else:
                print(f"Warning: Entry {i+1} missing query or documents", flush=True)
                rewritten_queries.append(query)  # Keep original if no documents
        
        # Process in batches
        for i in range(0, len(all_prompts), batch_size):
            batch = all_prompts[i:i + batch_size]
            batch_indices = valid_indices[i:i + batch_size]
            
            print(f"Processing rewrite batch {i//batch_size + 1}: queries {i+1}-{min(i+batch_size, len(all_prompts))}", flush=True)
            
            try:
                # Generate rewritten queries for the batch
                outputs = self.llm.generate(batch, self.sampling_params)
                
                # Process responses
                for j, (output, orig_idx) in enumerate(zip(outputs, batch_indices)):
                    rewritten = output.outputs[0].text.strip()
                    
                    # Clean up the rewritten query
                    rewritten = rewritten.replace("Rewritten Query:", "").strip()
                    rewritten = rewritten.replace("Expanded Query:", "").strip()
                    
                    # Store result at correct index
                    while len(rewritten_queries) <= orig_idx:
                        rewritten_queries.append("")
                    rewritten_queries[orig_idx] = rewritten
                    
                    original = entries[orig_idx].get("question", "")[:50]
                    print(f"  ✓ Rewritten query {orig_idx+1}: {original}...", flush=True)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}", flush=True)
                # Keep original queries for failed batch
                for idx in batch_indices:
                    while len(rewritten_queries) <= idx:
                        rewritten_queries.append("")
                    rewritten_queries[idx] = entries[idx].get("question", "")
        
        # Fill in any missing entries with original queries
        for i, entry in enumerate(entries):
            if i >= len(rewritten_queries):
                rewritten_queries.append(entry.get("question", ""))
            elif not rewritten_queries[i]:
                rewritten_queries[i] = entry.get("question", "")
        
        return rewritten_queries

def process_multiple_entries(rewriter: QueryRewriterVLLM, 
                           input_file: str, 
                           output_file: str,
                           rewrite_type: str = "comprehensive",
                           batch_size: int = 8,
                           use_batch_processing: bool = True):
    """Process all JSON entries in a JSONL file with query rewriting"""
    
    processed_entries = []
    
    # Count total lines for progress
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total entries to process: {total_lines}", flush=True)
    except:
        total_lines = "unknown"
    
    if use_batch_processing:
        # Load all entries first for batch processing
        all_entries = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    all_entries.append(data)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON, skipping line", flush=True)
                    continue
        
        print(f"Loaded {len(all_entries)} entries for batch processing", flush=True)
        
        # Batch rewrite all queries
        rewritten_queries = rewriter.rewrite_queries_batch(
            all_entries, rewrite_type, batch_size
        )
        
        # Create new entries with rewritten queries
        for entry, rewritten in zip(all_entries, rewritten_queries):
            # Create a new simplified entry
            new_entry = {
                "question": rewritten,  # Rewritten query goes in question field
                "original_query": entry.get("question", ""),  # Original query saved here
                "input": entry.get("input", ""),  # Keep input field if it exists
                "id": entry.get("id", 0),  # Keep id field
                "answers": entry.get("answers", []),  # Keep answers field if it exists
            }
            
            # Only add input_negative_ctxs if it exists in the original
            if "input_negative_ctxs" in entry:
                new_entry["input_negative_ctxs"] = entry["input_negative_ctxs"]
            
            processed_entries.append(new_entry)
    
    else:
        # Process entries one by one (non-batch mode)
        with open(input_file, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line = line.strip()
                line_num += 1
                
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    print(f"\n--- Processing entry {line_num}/{total_lines} ---", flush=True)
                    print(f"Original query: {data.get('question', '')[:100]}...", flush=True)
                    
                    # Get rewritten query
                    query = data.get("question", "")
                    documents = data.get("ctxs", [])
                    
                    if query and documents:
                        prompt = rewriter.create_rewrite_prompt(query, documents, rewrite_type)
                        outputs = rewriter.llm.generate([prompt], rewriter.sampling_params)
                        rewritten = outputs[0].outputs[0].text.strip()
                        rewritten = rewritten.replace("Rewritten Query:", "").strip()
                        rewritten = rewritten.replace("Expanded Query:", "").strip()
                    else:
                        rewritten = query
                    
                    # Create new simplified entry
                    new_entry = {
                        "question": rewritten,
                        "original_query": query,
                        "input": data.get("input", ""),
                        "id": data.get("id", 0),
                        "answers": data.get("answers", []),
                    }
                    
                    if "input_negative_ctxs" in data:
                        new_entry["input_negative_ctxs"] = data["input_negative_ctxs"]
                    
                    processed_entries.append(new_entry)
                    
                    print(f"Rewritten query: {rewritten[:100]}...", flush=True)
                    
                    # Optional: Save progress periodically
                    if line_num % 10 == 0:
                        temp_output = output_file.replace('.json', f'_temp_{line_num}.json')
                        with open(temp_output, 'w', encoding='utf-8') as temp_f:
                            for entry in processed_entries:
                                temp_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        print(f"Progress saved to {temp_output}", flush=True)
                    
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON on line {line_num}", flush=True)
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}", flush=True)
                    continue
    
    # Save final results as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n=== Rewriting Statistics ===")
    print(f"Total entries processed: {len(processed_entries)}")
    
    # Calculate average query length change
    original_lengths = [len(e.get("original_query", "")) for e in processed_entries]
    rewritten_lengths = [len(e.get("question", "")) for e in processed_entries]
    avg_original = sum(original_lengths) / len(original_lengths) if original_lengths else 0
    avg_rewritten = sum(rewritten_lengths) / len(rewritten_lengths) if rewritten_lengths else 0
    
    print(f"Average original query length: {avg_original:.1f} characters")
    print(f"Average rewritten query length: {avg_rewritten:.1f} characters")
    print(f"Average length change: {avg_rewritten - avg_original:+.1f} characters")
    
    return processed_entries

def main():
    # Configuration
    INPUT_FILE = "contriever_base_finetuned_t200_MAX12.json"  # Your filtered gold documents file
    OUTPUT_FILE = "qampari_contriever_rewritten_queries_expansion.json"
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
    
    # Rewriting configuration
    REWRITE_TYPE = "expansion" 
    BATCH_SIZE = 16  
    USE_BATCH_PROCESSING = True  
    MAX_CONTEXT_LENGTH = 8096
    
    print("=== Query Rewriting with vLLM ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Rewrite type: {REWRITE_TYPE}")
    print(f"Batch processing: {USE_BATCH_PROCESSING}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Initialize rewriter
    rewriter = QueryRewriterVLLM(MODEL_NAME, MAX_CONTEXT_LENGTH)
    
    try:
        # Process entries
        processed_entries = process_multiple_entries(
            rewriter, 
            INPUT_FILE, 
            OUTPUT_FILE,
            REWRITE_TYPE,
            BATCH_SIZE,
            USE_BATCH_PROCESSING
        )
        
        print(f"\n✓ Successfully processed {len(processed_entries)} entries")
        print(f"✓ Results saved to: {OUTPUT_FILE}")
        
        # Show sample results
        if processed_entries:
            print("\n=== Sample Results ===")
            for i in range(min(3, len(processed_entries))):
                entry = processed_entries[i]
                print(f"\nExample {i+1}:")
                print(f"Original: {entry.get('original_query', '')[:150]}...")
                print(f"Rewritten: {entry.get('question', '')[:150]}...")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()