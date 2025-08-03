import json
import os
def read_json_file(file_path):
    """Read JSON file and return list of dictionaries."""
    with open(file_path, 'r') as file:
        return json.load(file)

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
        
        if 'input_negative_ctxs' in item and item['input_negative_ctxs']:
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
            "ctxs": item.get('ctxs', []),
            "input_negative_ctxs": item.get('input_negative_ctxs', [])
        }
        
        converted_data.append(new_item)
    
    write_json_file(converted_data, output_file)
    
    print(f"Converted {len(converted_data)} items")
    print(f"Output saved to: {output_file}")
    
    
def main():
    input_file = "two_stage/gold_top3/qampari_standard_only_rand_gold_ER_3.json"
    output_file = "two_stage/gold_top3/qampari_standard_only_rand_gold_ER_3_concat.json"  

    input_file = "two_stage/vanilla_top3/qampari_standard_only_rand_vanilla_0123.json"
    output_file = "two_stage/vanilla_top3/qampari_standard_only_rand_vanilla_0123_concat.json"  
    
    concat_documents_to_question(input_file, output_file)

    #os.remove(output_file)

if __name__ == "__main__":

    main()