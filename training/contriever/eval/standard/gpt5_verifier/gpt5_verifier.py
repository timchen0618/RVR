import json
import openai
from typing import List, Dict, Any
import time
import argparse

class DocumentVerifier:
    def __init__(self, api_key: str, model: str = "gpt-4"):  # Fixed model name
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,  # Increased from 10
                temperature=0
            )
            
            answer = response.choices[0].message.content.strip().upper()
            print(f"Answer from model is: {answer}")
            return answer == "YES"
            
        except Exception as e:
            print(f"Error verifying document '{document.get('title', 'Unknown')}': {e}")
            return False
    
    def filter_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query:
            print("Warning: No question found in data")
            return data
        
        print(f"Processing query: {query}")
        print(f"Evaluating {len(documents)} documents...")
        
        gold_documents = []
        
        for i, doc in enumerate(documents):
            print(f"Checking document {i+1}/{len(documents)}: {doc.get('title', 'No title')[:50]}...")
            
            if self.verify_document(query, doc):
                gold_documents.append(doc)
                print(f"  ✓ GOLD - Document contains relevant answer")
            else:
                print(f"  ✗ FILTERED - Document does not contain relevant answer")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Create filtered data
        filtered_data = data.copy()
        filtered_data["ctxs"] = gold_documents
        
        print(f"\nFiltering complete:")
        print(f"Original documents: {len(documents)}")
        print(f"Gold documents: {len(gold_documents)}")
        print(f"Filtered out: {len(documents) - len(gold_documents)}")
        
        return filtered_data

def load_first_json_entry(filename: str) -> Dict[str, Any]:
    """Load the first JSON object from a JSONL file"""
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
                print(f"\n--- Processing entry {line_num} ---")
                filtered_data = verifier.filter_documents(data)
                processed_entries.append(filtered_data)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")
                continue
    
    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return processed_entries

def main():
    parser = argparse.ArgumentParser(description='Filter documents using OpenAI API verification')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Configuration
    INPUT_FILE = "qampari_corpus_base_finetuned_only_random_inp0_t100.json"
    OUTPUT_FILE = "filtered_documents.json"
    PROCESS_MULTIPLE = False  # Set to True to process all entries
    
    # Initialize verifier
    verifier = DocumentVerifier(args.api_key)
    
    try:
        if PROCESS_MULTIPLE:
            # Process all entries in the JSONL file
            processed_entries = process_multiple_entries(verifier, INPUT_FILE, OUTPUT_FILE)
            print(f"\nProcessed {len(processed_entries)} entries and saved to: {OUTPUT_FILE}")
        else:
            # Process only the first entry for testing
            print("Processing first entry only (test mode)...")
            data = load_first_json_entry(INPUT_FILE)
            
            filtered_data = verifier.filter_documents(data)
            
            # Save filtered results as a single JSON object
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