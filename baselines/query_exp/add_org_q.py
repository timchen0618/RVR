import json
from tqdm import trange

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

org_q_dir = '/scratch/cluster/hungting/projects/Multi_Answer/Data/retrieval_outputs/wiki/contriever/raw_query_exp'

corpus = 'wiki'
retriever = 'dpr'
rootdir = f'/scratch/cluster/hungting/projects/Multi_Answer/Data/retrieval_outputs/{corpus}/{retriever}'
data_types = ['arguana_generated', 'kialo', 'opinionqa']

for data_type in data_types:
    data = read_jsonl(f'{rootdir}/{data_type}_1k_query_exp.jsonl')
    org_q_data = read_jsonl(f'{org_q_dir}/{data_type}_1k_query_exp.jsonl')

    assert len(data) == len(org_q_data), (len(data), len(org_q_data))
    for i in trange(len(data)):
        assert data[i]['question'].replace('‘', '\'').replace('’', '\'') == org_q_data[i]['question'].replace('‘', '\'').replace('’', '\''), (data[i]['question'], org_q_data[i]['question'], i)
        data[i]['org_q'] = org_q_data[i]['org_q']

    write_jsonl(f'{rootdir}/{data_type}_1k_query_exp.jsonl', data)