import csv
from src.utils.file_utils import read_jsonl, write_tsv
from pathlib import Path
import random

## Observe Retrieval Outputs

retrieve_dir = Path('/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/quest/reranked')
data_dir = Path('/scratch/cluster/hungting/projects/diverse_response/data/quest_data')
retriever = 'bm25'

# columns needed: question, answers (entity names for QUEST), retrieved documents
retrieved_outputs = read_jsonl(retrieve_dir / f'{retriever}_reranked_l0.9.jsonl')
data = read_jsonl(data_dir / 'dev_data.jsonl')
assert len(retrieved_outputs) == len(data)

outputs = [['question', 'answers', 'top10_documents', 'retrieved_document_1', 'retrieved_document_2', 'retrieved_document_3', 'retrieved_document_4', 'retrieved_document_5', 'retrieved_document_6', 'retrieved_document_7', 'retrieved_document_8', 'retrieved_document_9', 'retrieved_document_10']]
for inst, retrieved in zip(data, retrieved_outputs):
    answers = '[' + ', '.join(['\"' + ans + '\"' for ans in inst['docs']]) + ']'
    
    top10_docs = '[' + ', '.join(['\"' + d['title'] + '\"' for d in retrieved['ctxs'][:10]]) + ']'
    retrieved_documents = [d['title'] + ' || ' + d['text'] for d in retrieved['ctxs'][:10]]
    outputs.append([retrieved['question'], answers, top10_docs] + retrieved_documents)    
    
write_tsv(outputs, Path('data/csvs') / f'quest_{retriever}_reranked_l0.9.tsv')


# ## Observe 2nd Stage Training Inputs / Outputs
# data_dir = Path('/scratch/cluster/hungting/projects/diverse_response/data/quest_data')

# data = read_jsonl(data_dir / '2nd_stage/train_data.jsonl')
# outputs = [['input', 'positive_ctx', 'negative_ctx']]
# for inst in data:
#     outputs.append([inst['question'], random.choice(inst['positive_ctxs']), random.choice(inst['hard_negative_ctxs'])])

# write_tsv(outputs, Path('data/csvs') / f'quest_2nd_stage_training.tsv')