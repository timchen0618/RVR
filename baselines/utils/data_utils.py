import json

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


# corpus = 'wiki' # ['wiki', 'sphere', 'arguana']
# retriever = 'contriever' # ['contriever', 'google', 'dpr', 'bm25']

def collect_retrieval_results(rootdir, corpus, retriever, data_types, query_exp=False, rerank=False):
    if query_exp and rerank:
        raise ValueError("Cannot have both query_exp and rerank")
    all_data = []
    if retriever not in ['contriever', 'dpr', 'bm25', 'google_api', 'tart', 'nv', 'stella']:
        raise ValueError(f"Retriever {retriever} not supported")
        
    for data_type in data_types:
        if query_exp:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}/{data_type}_{retriever}_query_exp.jsonl')) 
        elif rerank:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}/{data_type}_{retriever}_reranked.jsonl')) 
        else:
            all_data.append(read_jsonl(f'{rootdir}/{data_type}/{data_type}_{retriever}.jsonl'))
    return all_data
    