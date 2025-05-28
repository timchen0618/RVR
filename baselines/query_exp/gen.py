from openai import OpenAI
import json
from pathlib import Path
import csv
from tqdm import tqdm

def pred_gpt4(client, prompt):
    messages = [{"role": "user", "content": prompt}]
        
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages
    )
    response = response.choices[0].message.content.strip()
    return response

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')

def write_jsonl_line(filename, data):
    with open(filename, 'a') as fout:
        fout.write(json.dumps(data) + '\n')

def form_prompt(instruction, input_text):
    return instruction.replace('[Question]', input_text)

def form_prompt_winput(instruction, question, documents):
    return instruction.replace('[Question]', question).replace('[Documents]', documents)

def write_tsv(filename, data):
    with open(filename, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerows(data)


instruction = open('instruction_keywords.txt', 'r').read()
instruction_winput = open('instruction_keywords_winput.txt', 'r').read()
input_file = '/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus.jsonl'

out_data = []
out_file = Path(input_file).stem + '_query_exp.jsonl'

data = read_jsonl(input_file)
client = OpenAI()

for inst in tqdm(data):
    prompt = form_prompt(instruction, inst['org_q'])
    input_docs = [c['title'] + ' ' + c['text'] for c in inst['input_negative_ctxs']]
    prompt_winput = form_prompt_winput(instruction_winput, inst['org_q'], '\n\n'.join(input_docs))
    
    # try:
    response = pred_gpt4(client, prompt)
    # except:
    #     print('skipped question', inst['question'])
    #     continue
    
    # try:
    response_winput = pred_gpt4(client, prompt_winput)
    # except:
    #     print('skipped question winput', inst['question'])
    #     continue
    
    inst['keywords'] = response.split(',')
    inst['keywords_winput'] = response_winput.split(',')
    out_data.append(inst)
    
    write_jsonl_line(out_file, inst)

# write_jsonl(out_file, data)