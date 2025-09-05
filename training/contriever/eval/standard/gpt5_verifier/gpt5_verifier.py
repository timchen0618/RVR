import json
import openai
from typing import List, Dict, Any
import time
import argparse
from openai import OpenAI
import os
import sys
import locale

# Force UTF-8 environment variables at the very top
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

# Set locale to UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        pass  # Continue anyway

# Reconfigure stdout/stderr for UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class DocumentVerifier:
    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def clean_text(self, text):
        """More aggressive text cleaning for Unicode issues"""
        if isinstance(text, str):
            # Replace problematic Unicode quotes with standard ones
            text = text.replace('\u201c', '"').replace('\u201d', '"')
            text = text.replace('\u2018', "'").replace('\u2019', "'")
            text = text.replace('\u2013', '-').replace('\u2014', '--')  # en/em dashes
            text = text.replace('\u2026', '...')  # ellipsis
            text = text.replace('\u00a0', ' ')  # non-breaking space
            
            # More aggressive approach: encode to ASCII and ignore errors
            # This will strip out any remaining problematic characters
            try:
                # First try to encode/decode safely
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                # Then convert to ASCII-safe version
                text = text.encode('ascii', errors='ignore').decode('ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback: remove all non-ASCII characters
                text = ''.join(char for char in text if ord(char) < 128)
                
        return text
    
    def verify_document(self, query: str, document: Dict[str, Any]) -> bool:
        # Clean all text inputs aggressively
        title = self.clean_text(document.get('title', 'No title'))
        text = self.clean_text(document.get('text', ''))
        query = self.clean_text(query)

        # Create verification prompt with cleaned text
        prompt = f"""You are a document relevance verifier. Your task is to determine if a given document contains a direct answer to a specific question.

Question: {query}

Document Title: {title}
Document Text: {text}

Instructions:
- Answer with ONLY "YES" if the document directly answers the question or contains clear, specific information that answers the question
- Answer with ONLY "NO" if the document does not contain a direct answer, is only tangentially related, or does not provide the specific information asked for
- Be strict in your evaluation - the document should contain factual information that directly addresses the query

Your answer (YES or NO):"""

        try:
            # Ensure prompt is ASCII-safe before sending
            prompt = self.clean_text(prompt)
            
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            answer = response.output_text.strip().upper()
            print(f"Answer from model is: {answer}", flush=True)
            return answer == "YES"
            
        except Exception as e:
            print(f"Error verifying document '{self.clean_text(document.get('title', 'Unknown'))}': {e}", flush=True)
            return False
    
    def filter_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query:
            print("Warning: No question found in data", flush=True)
            return data
        
        # Clean query for display
        clean_query = self.clean_text(query)
        print(f"Processing query: {clean_query}", flush=True)
        print(f"Evaluating {len(documents)} documents...", flush=True)
        
        gold_documents = []
        
        for i, doc in enumerate(documents):
            doc_title = self.clean_text(doc.get('title', 'No title'))
            print(f"Checking document {i+1}/{len(documents)}: {doc_title[:50]}...", flush=True)
            
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

def process_multiple_entries(verifier, input_file: str, output_file: str):
    """Process all JSON entries in a JSONL file"""
    processed_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                print(f"\n--- Processing entry {line_num} ---", flush=True)
                filtered_data = verifier.filter_documents(data)
                processed_entries.append(filtered_data)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}", flush=True)
                continue
    
    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=True) + '\n')  # Changed to ensure_ascii=True
    
    return processed_entries

def main():
    parser = argparse.ArgumentParser(description='Filter documents using OpenAI API verification')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Configuration
    INPUT_FILE = "qampari_corpus_base_finetuned_only_random_inp0_t100.json"
    OUTPUT_FILE = "qampari_base_gpt5_nano_all_gold_docs_100.json"
    PROCESS_MULTIPLE = False  # Set to True to process all entries
    
    # Initialize verifier
    verifier = DocumentVerifier(args.api_key)
    
    try:
        if PROCESS_MULTIPLE:
            # Process all entries in the JSONL file
            print("About to process multiple entries", flush=True)
            processed_entries = process_multiple_entries(verifier, INPUT_FILE, OUTPUT_FILE)
            print(f"\nProcessed {len(processed_entries)} entries and saved to: {OUTPUT_FILE}", flush=True)
        else:
            # Process only the first entry for testing
            print("Processing first entry only (test mode)...", flush=True)
            data = load_first_json_entry(INPUT_FILE)
            
            filtered_data = verifier.filter_documents(data)
            
            # Save filtered results as a single JSON object
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=True)  # Changed to ensure_ascii=True
            
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