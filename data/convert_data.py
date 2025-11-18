import json

# Input and output file paths
INPUT_FILE = "/scratch/dq2024/diverse_retriever/data/quest_test.jsonl"
OUTPUT_FILE_1 = "/scratch/dq2024/diverse_retriever/data/dev_data_gt_quest.jsonl"
OUTPUT_FILE_2 = "/scratch/dq2024/diverse_retriever/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_webqsp_concat_query_inp0.json"

def convert_to_format1(input_data):
    """Convert to first format with question_text and answer_list"""
    results = []
    qid_counter = 0
    
    with open(input_data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                
                # Skip if no documents
                if not obj.get('docs') or len(obj['docs']) == 0:
                    continue
                
                # Create answer_list from docs
                answer_list = []
                for doc in obj['docs']:
                    answer_list.append({
                        "answer_text": doc,
                        "aid": f"m.{qid_counter:05d}",
                        "aliases": [doc],
                        "answer_url": "",
                        "proof": []
                    })
                
                # Create output object
                output_obj = {
                    "question_text": obj['query'],
                    "answer_list": answer_list,
                    "qid": f"WebQTest-{qid_counter}",
                    "ground_truths": []
                }
                
                results.append(output_obj)
                qid_counter += 1
    
    return results

def convert_to_format2(input_data):
    """Convert to second format with question and input fields"""
    results = []
    id_counter = 0
    
    with open(input_data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                
                # Skip if no documents
                if not obj.get('docs') or len(obj['docs']) == 0:
                    continue
                
                # Create output object
                output_obj = {
                    "question": obj['query'],
                    "input": obj['query'],
                    "id": id_counter,
                    "answers": [""],
                    "ctxs": [],
                    "input_negative_ctxs": []
                }
                
                results.append(output_obj)
                id_counter += 1
    
    return results

def main():
    # Convert to format 1 (JSONL output)
    print(f"Reading from {INPUT_FILE}...")
    format1_data = convert_to_format1(INPUT_FILE)
    
    print(f"Writing format 1 to {OUTPUT_FILE_1}...")
    with open(OUTPUT_FILE_1, 'w', encoding='utf-8') as f:
        for obj in format1_data:
            f.write(json.dumps(obj) + '\n')
    
    print(f"Format 1: Wrote {len(format1_data)} records to {OUTPUT_FILE_1}")
    
    # Convert to format 2 (JSON array output)
    format2_data = convert_to_format2(INPUT_FILE)
    
    print(f"Writing format 2 to {OUTPUT_FILE_2}...")
    with open(OUTPUT_FILE_2, 'w', encoding='utf-8') as f:
        json.dump(format2_data, f, indent=4)
    
    print(f"Format 2: Wrote {len(format2_data)} records to {OUTPUT_FILE_2}")
    print("Conversion complete!")

if __name__ == "__main__":
    main()