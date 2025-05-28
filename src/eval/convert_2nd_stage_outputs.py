from src.utils.file_utils import read_jsonl, write_jsonl

input_path = '/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_wiki_2nd_stage/quest_dev_2nd_stage.json'
preds = read_jsonl(input_path)
raw_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/quest_data/2nd_stage/dev_data.jsonl')
newq2orgq = {}
for inst in raw_data:
    newq2orgq[inst['question']] = inst['org_q']

assert len(preds) == len(raw_data)

for pred, inst in zip(preds, raw_data):
    pred['question'] = newq2orgq[pred['question']]


write_jsonl(preds, input_path)