import json
import openai
from typing import List, Dict, Any
import time
import argparse
from openai import OpenAI
import os
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

class DocumentVerifier:
    def __init__(self, api_key: str, model: str = "gpt-5-nano", max_workers: int = 10):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.rate_limit_delay = 0.05  # Reduced from 0.1
        self.batch_size = 5  # Process documents in batches
        
    def verify_document_batch(self, query: str, documents: List[Dict[str, Any]]) -> List[bool]:
        """Verify multiple documents in a single API call using batch processing"""
        if not documents:
            return []
        
        query = query.encode('ascii', 'ignore').decode('ascii')
        
        # Create a batch prompt for multiple documents
        batch_prompt = f"""You are a document relevance verifier. Your task is to determine if given documents contain direct answers to a specific question.

Question: {query}

Documents to evaluate:
"""
        
        for i, doc in enumerate(documents):
            # Clean the text to handle encoding issues
            title = doc.get('title', 'No title').encode('ascii', 'ignore').decode('ascii')
            text = doc.get('text', '')[:1000].encode('ascii', 'ignore').decode('ascii')
            
            batch_prompt += f"""
Document {i+1}:
Title: {title}
Text: {text}...  # Truncate for efficiency

"""
        
        batch_prompt += """
Instructions:
- For each document, answer with ONLY "YES" if it directly answers the question or "NO" if it doesn't
- Provide your answers as a comma-separated list in the exact order of the documents
- Example format: YES,NO,YES,NO,YES
- Be strict in your evaluation - documents should contain factual information that directly addresses the query

Your answers (comma-separated):"""

       # try:
        response = self.client.responses.create(
            model=self.model,
            input=batch_prompt,
        )
        answer = response.output_text.strip().upper()
        
        # Parse the comma-separated response
        answers = [ans.strip() == "YES" for ans in answer.split(',')]
            
            # Ensure we have the right number of answers
            # if len(answers) != len(documents):
            #     print(f"Warning: Expected {len(documents)} answers, got {len(answers)}. Falling back to individual verification.", flush=True)
            #     return [self.verify_document_individual(query, doc) for doc in documents]
            
        return answers
            
        # except Exception as e:
        #     print(f"Error in batch verification: {e}. Falling back to individual verification.", flush=True)
        #     return [self.verify_document_individual(query, doc) for doc in documents]

    def verify_document_individual(self, query: str, document: Dict[str, Any]) -> bool:
        """Fallback method for individual document verification"""
        
        prompt = f"""You are a document relevance verifier. Your task is to determine if a given document contains a direct answer to a specific question.

Question: {query}

Document Title: {document.get('title', 'No title')}
Document Text: {document.get('text', '')}

Instructions:
- Answer with ONLY "YES" if the document directly answers the question or contains clear, specific information that answers the question
- Answer with ONLY "NO" if the document does not contain a direct answer, is only tangentially related, or does not provide the specific information asked for
- Be strict in your evaluation - the document should contain factual information that directly addresses the query

Your answer (YES or NO):"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            answer = response.output_text.strip().upper()
            return answer == "YES"
            
        except Exception as e:
            print(f"Error verifying document '{document.get('title', 'Unknown')}': {e}", flush=True)
            return False

    def verify_document(self, query: str, document: Dict[str, Any]) -> bool:
        """Wrapper method to maintain compatibility"""
        return self.verify_document_individual(query, document)
    
    def filter_documents_optimized(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized version using batch processing"""
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query:
            print("Warning: No question found in data", flush=True)
            return data
        
        print(f"Processing query: {query}", flush=True)
        print(f"Evaluating {len(documents)} documents using batch processing...", flush=True)
        
        gold_documents = []
        
        # Process documents in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}: documents {i+1}-{min(i+self.batch_size, len(documents))}", flush=True)
            
            # Get batch results
            batch_results = self.verify_document_batch(query, batch)
            
            # Add gold documents from this batch
            for j, is_gold in enumerate(batch_results):
                doc_idx = i + j
                if is_gold:
                    gold_documents.append(documents[doc_idx])
                    print(f"  ✓ GOLD - Document {doc_idx+1}: {documents[doc_idx].get('title', 'No title')[:50]}...", flush=True)
                else:
                    print(f"  ✗ FILTERED - Document {doc_idx+1}: {documents[doc_idx].get('title', 'No title')[:50]}...", flush=True)
            
            # Reduced delay between batches
            if i + self.batch_size < len(documents):
                time.sleep(self.rate_limit_delay)
        
        # Create filtered data
        filtered_data = data.copy()
        filtered_data["ctxs"] = gold_documents
        
        print(f"\nBatch filtering complete:", flush=True)
        print(f"Original documents: {len(documents)}", flush=True)
        print(f"Gold documents: {len(gold_documents)}", flush=True)
        print(f"Filtered out: {len(documents) - len(gold_documents)}", flush=True)
        
        return filtered_data
    
    def filter_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Use optimized batch processing by default"""
        return self.filter_documents_optimized(data)

def load_first_json_entry(filename: str) -> Dict[str, Any]:
    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("File is empty or first line is blank")
        return json.loads(first_line)

def load_checkpoint(checkpoint_file: str) -> tuple[List[Dict[str, Any]], int]:
    """Load processed entries and last processed line number from checkpoint"""
    if not os.path.exists(checkpoint_file):
        return [], 0
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            processed_entries = checkpoint_data.get('processed_entries', [])
            last_line = checkpoint_data.get('last_processed_line', 0)
            print(f"Loaded checkpoint: {len(processed_entries)} entries processed, resuming from line {last_line + 1}", flush=True)
            return processed_entries, last_line
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting fresh.", flush=True)
        return [], 0

def save_checkpoint(checkpoint_file: str, processed_entries: List[Dict[str, Any]], last_line: int):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'processed_entries': processed_entries,
        'last_processed_line': last_line,
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(processed_entries)
    }
    
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        print(f"Checkpoint saved: {len(processed_entries)} entries processed", flush=True)
    except Exception as e:
        print(f"Error saving checkpoint: {e}", flush=True)

def process_multiple_entries(verifier, input_file: str, output_file: str):
    """Process all JSON entries in a JSONL file with optimizations"""
    
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    processed_entries, start_line = load_checkpoint(checkpoint_file)
    
    # Count total lines for progress
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total lines to process: {total_lines}", flush=True)
    except:
        total_lines = "unknown"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Skip already processed lines
            for _ in range(start_line):
                f.readline()
            
            line_num = start_line
            for line in f:
                line = line.strip()
                line_num += 1
                
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    print(f"\n--- Processing entry {line_num} ({len(processed_entries)+1} total processed) ---", flush=True)
                    
                    # Use optimized filtering
                    filtered_data = verifier.filter_documents_optimized(data)
                    processed_entries.append(filtered_data)
                    
                    # Save checkpoint every 3 entries (more frequent saves)
                    if len(processed_entries) % 3 == 0:
                        save_checkpoint(checkpoint_file, processed_entries, line_num)
                        
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON on line {line_num}", flush=True)
                    continue
                except KeyboardInterrupt:
                    print(f"\nInterrupted! Saving checkpoint...", flush=True)
                    save_checkpoint(checkpoint_file, processed_entries, line_num - 1)
                    raise
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}", flush=True)
                    save_checkpoint(checkpoint_file, processed_entries, line_num)
                    continue
    
    except Exception as e:
        print(f"Fatal error: {e}", flush=True)
        save_checkpoint(checkpoint_file, processed_entries, line_num if 'line_num' in locals() else start_line)
        raise
    
    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Clean up checkpoint on successful completion
    try:
        os.remove(checkpoint_file)
        print(f"Processing complete! Checkpoint file removed.", flush=True)
    except:
        pass
    
    return processed_entries

def main():
    parser = argparse.ArgumentParser(description='Filter documents using OpenAI API verification')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of documents to process in each batch (default: 5)')
    parser.add_argument('--rate-limit-delay', type=float, default=0.05, help='Delay between API calls (default: 0.05)')
    
    args = parser.parse_args()
    
    # Configuration
    INPUT_FILE = "/scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t100.json"
    OUTPUT_FILE = "/scratch/dq2024/diverse_retriever/inference/gpt_verifier/qampari_infly_base_gpt5_nano_all_gold_docs_100.json"
    PROCESS_MULTIPLE = True
    
    # Initialize verifier with optimizations
    verifier = DocumentVerifier(
        args.api_key, 
        max_workers=10
    )
    verifier.batch_size = args.batch_size
    verifier.rate_limit_delay = args.rate_limit_delay
    
    try:
        if PROCESS_MULTIPLE:
            print("About to process multiple entries with optimizations", flush=True)
            processed_entries = process_multiple_entries(verifier, INPUT_FILE, OUTPUT_FILE)
            print(f"\nProcessed {len(processed_entries)} entries and saved to: {OUTPUT_FILE}", flush=True)
        else:
            print("Processing first entry only (test mode)...", flush=True)
            data = load_first_json_entry(INPUT_FILE)
            
            filtered_data = verifier.filter_documents(data)
            
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nFiltered data saved to: {OUTPUT_FILE}", flush=True)
        
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