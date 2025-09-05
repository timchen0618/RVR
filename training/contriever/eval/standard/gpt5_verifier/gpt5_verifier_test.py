import json
import openai
from typing import List, Dict, Any
import time
import argparse
from openai import OpenAI
import os
from datetime import datetime

class DocumentVerifier:
    def __init__(self, api_key: str, model: str = "gpt-5-nano"):  # Fixed model name
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def verify_document(self, query: str, document: Dict[str, Any]) -> bool:
        # Create verification prompt
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
            print(f"Answer from model is: {answer}", flush=True)
            return answer == "YES"
            
        except Exception as e:
            print(f"Error verifying document '{document.get('title', 'Unknown')}': {e}", flush=True)
            return False
    
    def filter_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query:
            print("Warning: No question found in data", flush=True)
            return data
        
        print(f"Processing query: {query}", flush=True)
        print(f"Evaluating {len(documents)} documents...", flush=True)
        
        gold_documents = []
        
        for i, doc in enumerate(documents):
            print(f"Checking document {i+1}/{len(documents)}: {doc.get('title', 'No title')[:50]}...", flush=True)
            
            if self.verify_document(query, doc):
                gold_documents.append(doc)
                print(f"  ✓ GOLD - Document contains relevant answer", flush=True)
            else:
                print(f"  ✗ FILTERED - Document does not contain relevant answer", flush=True)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Create filtered data
        filtered_data = data.copy()
        filtered_data["ctxs"] = gold_documents
        
        print(f"\nFiltering complete:", flush=True)
        print(f"Original documents: {len(documents)}", flush=True)
        print(f"Gold documents: {len(gold_documents)}", flush=True)
        print(f"Filtered out: {len(documents) - len(gold_documents)}", flush=True)
        
        return filtered_data

def load_first_json_entry(filename: str) -> Dict[str, Any]:
    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("File is empty or first line is blank")
        return json.loads(first_line)

# NEW CHECKPOINT FUNCTIONS
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
    """Process all JSON entries in a JSONL file"""
    
    # NEW: Setup checkpoint file
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    processed_entries, start_line = load_checkpoint(checkpoint_file)
    
    # NEW: Count total lines for progress
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total lines to process: {total_lines}", flush=True)
    except:
        total_lines = "unknown"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # NEW: Skip already processed lines
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
                    filtered_data = verifier.filter_documents(data)
                    processed_entries.append(filtered_data)
                    
                    # NEW: Save checkpoint every 5 entries
                    if len(processed_entries) % 5 == 0:
                        save_checkpoint(checkpoint_file, processed_entries, line_num)
                        
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON on line {line_num}", flush=True)
                    continue
                except KeyboardInterrupt:
                    # NEW: Handle interruption gracefully
                    print(f"\nInterrupted! Saving checkpoint...", flush=True)
                    save_checkpoint(checkpoint_file, processed_entries, line_num - 1)
                    raise
                except Exception as e:
                    # NEW: Save checkpoint on error and continue
                    print(f"Error processing line {line_num}: {e}", flush=True)
                    save_checkpoint(checkpoint_file, processed_entries, line_num)
                    continue
    
    except Exception as e:
        # NEW: Save checkpoint on fatal error
        print(f"Fatal error: {e}", flush=True)
        save_checkpoint(checkpoint_file, processed_entries, line_num if 'line_num' in locals() else start_line)
        raise
    
    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # NEW: Clean up checkpoint on successful completion
    try:
        os.remove(checkpoint_file)
        print(f"Processing complete! Checkpoint file removed.", flush=True)
    except:
        pass
    
    return processed_entries

def main():
    parser = argparse.ArgumentParser(description='Filter documents using OpenAI API verification')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Configuration
    INPUT_FILE = "qampari_corpus_base_finetuned_only_random_inp0_t100.json"
    OUTPUT_FILE = "test.json"
    PROCESS_MULTIPLE = True  # Set to True to process all entries
    
    # Initialize verifier
    verifier = DocumentVerifier(args.api_key)
    
    try:
        if PROCESS_MULTIPLE:
            # Process all entries in the JSONL file
            print("About to process multuple entires", flush=True)
            processed_entries = process_multiple_entries(verifier, INPUT_FILE, OUTPUT_FILE)
            print(f"\nProcessed {len(processed_entries)} entries and saved to: {OUTPUT_FILE}", flush=True)
        else:
            # Process only the first entry for testing
            print("Processing first entry only (test mode)...", flush=True)
            data = load_first_json_entry(INPUT_FILE)
            
            filtered_data = verifier.filter_documents(data)
            
            # Save filtered results as a single JSON object
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