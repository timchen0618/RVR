import json
from tqdm import tqdm

# Read the JSONL file
with open('data/qampari_data/2nd_stage/dev_data.jsonl', 'r') as jsonl_file:
    data = [json.loads(line) for line in jsonl_file]

question_str = 'question' # 'question_text' for qampari, 'query' for quest, 'question' for 2nd stage data
# Modify the data format and field names
modified_data = []
for item in tqdm(data):
    modified_item = {
        'question': item[question_str],
        'input': item[question_str],
        'id': item['qid'] if 'qid' in item else len(modified_data),
        'answers': [''],
        'ctxs': ['']
    }
    modified_data.append(modified_item)

# Write the modified data to JSON file
with open('data/qampari_data/nqformat_data/qampari_dev_2nd_stage.json', 'w') as json_file:
    json.dump(modified_data, json_file)
