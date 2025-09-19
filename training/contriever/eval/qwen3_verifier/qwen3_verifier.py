import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

class DocumentVerifier:
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model loaded successfully!")
        print(f"Model device: {self.model.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
    def verify_document(self, query: str, document: Dict[str, Any]) -> bool:
        """Verify a single document for relevance to the query"""
        
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
            # Prepare messages for chat template
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=10,  # Short response expected
                do_sample=False,    # Deterministic output
                temperature=0.1,    # Low temperature for consistency
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated response
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Parse response
            answer = response.upper().strip()
            print(f"Model response: '{response}' -> Parsed: '{answer}'", flush=True)
            
            # Check if response contains YES
            return "YES" in answer and "NO" not in answer
            
        except Exception as e:
            print(f"Error verifying document '{document.get('title', 'Unknown')}': {e}", flush=True)
            return False

    def filter_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter documents by processing them one by one"""
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query:
            print("Warning: No question found in data", flush=True)
            return data
        
        print(f"Processing query: {query}", flush=True)
        print(f"Evaluating {len(documents)} documents individually...", flush=True)
        
        gold_documents = []
        
        # Process each document individually
        for i, document in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}: {document.get('title', 'No title')[:50]}...", flush=True)
            
            is_gold = self.verify_document(query, document)
            
            if is_gold:
                gold_documents.append(document)
                print(f"  ✓ GOLD - Document {i+1}", flush=True)
            else:
                print(f"  ✗ FILTERED - Document {i+1}", flush=True)
        
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
    
    # Count total lines for progress
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total lines to process: {total_lines}", flush=True)
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
                print(f"\n--- Processing entry {line_num} ---", flush=True)
                
                # Filter documents individually
                filtered_data = verifier.filter_documents(data)
                processed_entries.append(filtered_data)
                
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}", flush=True)
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}", flush=True)
                continue
    
    # Save as JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return processed_entries

def main():
    # Configuration
    INPUT_FILE = "/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/all_golds/qampari_corpus_base_finetuned_only_random_inp0_t100.json"
    OUTPUT_FILE = "qampari_base_qwen3_all_gold_docs_100.json"
    #MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    MODEL_NAME = "Qwen/Qwen3-8B"
    PROCESS_MULTIPLE = True
    
    # Initialize verifier
    verifier = DocumentVerifier(MODEL_NAME)
    
    try:
        if PROCESS_MULTIPLE:
            print("Processing multiple entries with Qwen3 model", flush=True)
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