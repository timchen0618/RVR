import json
import os
def read_json_file(file_path):
    import json
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    # Try parsing as regular JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try parsing as JSONL
        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data

def write_json_file(data, file_path):
    """Write data to JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def concat_documents_to_question(input_file, output_file):
    """
    Convert JSON format by concatenating input_negative_ctxs text to question.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    data = read_json_file(input_file)
    
    converted_data = []
    
    for item in data:
        base_question = f"Question: {item['question']}"
        
        if 'ctxs' in item and item['ctxs']:
            documents_text = "\n\nDocuments: "
            
            for i, doc in enumerate(item['ctxs']):
                doc_text = f"{doc['title']} {doc['text']}"
                if i == 0:
                    documents_text += doc_text
                else:
                    documents_text += f"\n{doc_text}"
            
            concatenated_text = base_question + documents_text
        elif 'input_negative_ctxs' in item and item['input_negative_ctxs']:
            documents_text = "\n\nDocuments: "
            
            for i, doc in enumerate(item['input_negative_ctxs']):
                doc_text = f"{doc['title']} {doc['text']}"
                if i == 0:
                    documents_text += doc_text
                else:
                    documents_text += f"\n{doc_text}"
            
            concatenated_text = base_question + documents_text
        else:
            concatenated_text = base_question
        
        new_item = {
            "question": concatenated_text,
            "input": concatenated_text,
            "id": item['id'],
            "answers": item['answers'],
            "ctxs": [],
            "input_negative_ctxs": []
        }
        
        converted_data.append(new_item)
    
    write_json_file(converted_data, output_file)
    
    print(f"Converted {len(converted_data)} items")
    print(f"Concatenated output saved to: {output_file}", flush=True)
    
    
def main():
    input_file = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/input_files/qampari_finetuned_base_t200_promote-qwen3_0-3_unformat_round_two.json"
    output_file = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/input_files/qampari_finetuned_base_t200_promote-qwen3_0-3_concat_round_two.json"
    concat_documents_to_question(input_file, output_file)

    

if __name__ == "__main__":

    main()
