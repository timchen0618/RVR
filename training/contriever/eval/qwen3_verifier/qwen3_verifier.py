import json
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple

class DocumentVerifierVLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        print(f"Loading vLLM model: {model_name}")
        
        # Initialize vLLM with optimized settings
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=4096,  # Adjust based on your needs
            trust_remote_code=True,
            dtype="auto"
        )
        
        # Set up sampling parameters to match original code exactly
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Make it deterministic like do_sample=False
            max_tokens=10,
            # Removed stop tokens to match original behavior
            skip_special_tokens=True
        )
        
        print("vLLM model loaded successfully!")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
    def create_verification_prompt(self, query: str, document: Dict[str, Any]) -> str:
        """Create verification prompt for a document"""
        return f"""You are a document relevance verifier. Your task is to determine if a given document contains a direct answer to a specific question.

Question: {query}

Document Title: {document.get('title', 'No title')}
Document Text: {document.get('text', '')}

Instructions:
- Answer with ONLY "YES" if the document directly answers the question or contains clear, specific information that answers the question
- Answer with ONLY "NO" if the document does not contain a direct answer, is only tangentially related, or does not provide the specific information asked for
- Be strict in your evaluation - the document should contain factual information that directly addresses the query

Your answer (YES or NO):"""

    def verify_documents_batch(self, query: str, documents: List[Dict[str, Any]], batch_size: int = 8) -> List[bool]:
        """Verify multiple documents in batches using vLLM"""
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}: documents {i+1}-{min(i+batch_size, len(documents))}")
            
            # Create prompts for the batch
            prompts = [self.create_verification_prompt(query, doc) for doc in batch]
            
            try:
                # Generate responses for the batch
                outputs = self.llm.generate(prompts, self.sampling_params)
                
                # Process responses
                batch_results = []
                for j, output in enumerate(outputs):
                    response = output.outputs[0].text.strip().upper()
                    is_gold = "YES" in response and "NO" not in response
                    batch_results.append(is_gold)
                    
                    doc_title = batch[j].get('title', 'No title')[:50]
                    status = "✓ GOLD" if is_gold else "✗ FILTERED"
                    print(f"  {status} - Document {i+j+1}: {doc_title}... (Response: '{output.outputs[0].text.strip()}')")
                
                results.extend(batch_results)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add False for all documents in failed batch
                results.extend([False] * len(batch))
        
        return results

    def filter_documents(self, data: Dict[str, Any], batch_size: int = 8) -> Dict[str, Any]:
        """Filter documents using batched processing"""
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query:
            print("Warning: No question found in data")
            return data
        
        if not documents:
            print("Warning: No documents found in data")
            return data
        
        print(f"Processing query: {query}")
        print(f"Evaluating {len(documents)} documents in batches of {batch_size}...")
        
        # Verify all documents in batches
        verification_results = self.verify_documents_batch(query, documents, batch_size)
        
        # Filter documents based on results
        gold_documents = [
            doc for doc, is_gold in zip(documents, verification_results) 
            if is_gold
        ]
        
        # Create filtered data
        filtered_data = data.copy()
        filtered_data["ctxs"] = gold_documents
        
        print(f"\nFiltering complete:")
        print(f"Original documents: {len(documents)}")
        print(f"Gold documents: {len(gold_documents)}")
        print(f"Filtered out: {len(documents) - len(gold_documents)}")
        
        return filtered_data

def load_first_json_entry(filename: str) -> Dict[str, Any]:
    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("File is empty or first line is blank")
        return json.loads(first_line)

def process_multiple_entries(verifier, input_file: str, output_file: str, batch_size: int = 8):
    """Process all JSON entries in a JSONL file with batched document verification"""
    
    processed_entries = []
    
    # Count total lines for progress
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total lines to process: {total_lines}")
    except:
        total_lines = "unknown"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            line_num += 1
            
            if not line:
                continue
                
            try:
                data = json.loads(line)
                print(f"\n--- Processing entry {line_num}/{total_lines} ---")
                
                # Filter documents using batched processing
                filtered_data = verifier.filter_documents(data, batch_size)
                processed_entries.append(filtered_data)
                
                # Optional: Save progress periodically
                if line_num % 10 == 0:
                    temp_output = output_file.replace('.json', f'_temp_{line_num}.json')
                    with open(temp_output, 'w', encoding='utf-8') as temp_f:
                        for entry in processed_entries:
                            temp_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    print(f"Progress saved to {temp_output}")
                
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Save final results as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return processed_entries

def main():
    # Configuration
    INPUT_FILE = "/scratch/dq2024/diverse_retriever/training/contriever/eval/base/qampari_corpus_base_only_random_inp0_t500.json"
    OUTPUT_FILE = "qampari_base_qwen3-30B_all_gold_docs_500_vllm.json"
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    PROCESS_MULTIPLE = True
    BATCH_SIZE = 16  # Increase batch size for better throughput with vLLM
    
    # Initialize verifier
    verifier = DocumentVerifierVLLM(MODEL_NAME)
    
    try:
        if PROCESS_MULTIPLE:
            print(f"Processing multiple entries with vLLM (batch size: {BATCH_SIZE})")
            processed_entries = process_multiple_entries(verifier, INPUT_FILE, OUTPUT_FILE, BATCH_SIZE)
            print(f"\nProcessed {len(processed_entries)} entries and saved to: {OUTPUT_FILE}")
        else:
            print("Processing first entry only (test mode)...")
            data = load_first_json_entry(INPUT_FILE)
            
            filtered_data = verifier.filter_documents(data, BATCH_SIZE)
            
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nFiltered data saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()