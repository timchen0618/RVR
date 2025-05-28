from utils.data_utils import collect_retrieval_results, write_jsonl, read_jsonl
import logging
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from sentence_transformers import SentenceTransformer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DocSimilarity:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.saved_similarities = [] # list of document similarities (100*100 matrices)
        self.num_docs = 100 + 3
    # @torch.no_grad()
    # def similarity(self, documents):
    #     # assert len(documents) == 100, len(documents)
    #     if len(documents) != 100:
    #         print('!=100', len(documents))

    #     self.saved_similarities.append(np.zeros((100, 100)))
    #     all_vecs = self.model.encode(documents, to_numpy=False)
    #     for i in range(len(documents)-1):

    #         target_vec = all_vecs[i:i+1]
    #         vecs = all_vecs[i+1:]
    #         cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #         output = cos(target_vec, vecs) # (len(documents)-i-1)
    #         self.saved_similarities[-1][i, i+1:len(documents)] = output.cpu().numpy()
    
    def similarity(self, documents):
        assert len(documents) == self.num_docs, len(documents)
        if len(documents) != self.num_docs:
            print('!=self.num_docs', len(documents))

        self.saved_similarities.append(np.zeros((self.num_docs, self.num_docs)))
        # all_vecs = self.model.encode(documents, to_numpy=False)
        
        query_embeddings = self.model.encode(documents, prompt_name="s2s_query")
        doc_embeddings = self.model.encode(documents)
        # (2, 1024) (2, 1024)

        similarities = self.model.similarity(query_embeddings, doc_embeddings)
        self.saved_similarities[-1] = similarities.cpu().numpy()
            
    def clear_similarity(self):
        self.saved_similarities = []

def main(args):

    tokenizer = None
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
    model.eval()
    # model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # data_types = ['qampari', 'quest']
    output_dir = '/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/qampari_2nd_stage'
    output_name = 'qampari_dev_2nd_stage_qampari_corpus_org_org_query_similarity.npy'
    
    retrieval_results = read_jsonl('/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_qampari/qampari_dev_2nd_stage_qampari_corpus_org_org_query.json')
    raw_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus.jsonl')
    assert len(retrieval_results) == len(raw_data), (len(retrieval_results), len(raw_data))
    doc_sim = DocSimilarity(model, tokenizer, device, args)

    for ret_inst, inst in tqdm(zip(retrieval_results, raw_data)):
        assert (ret_inst['question'] == inst['question'] or ret_inst['question'] == inst['org_q'])    
        
        documents = []
        
        for doc in inst['input_negative_ctxs']:
            if 'title' in doc:
                doc_text = doc['text'] + ' ' + doc['title']
            elif 'wikipedia_title' in doc:
                doc_text = doc['text'] + ' ' + doc['wikipedia_title']
            else:
                doc_text = doc['text']
            documents.append(doc_text)
            
        for doc in ret_inst['ctxs']:
            if 'title' in doc:
                doc_text = doc['text'] + ' ' + doc['title']
            elif 'wikipedia_title' in doc:
                doc_text = doc['text'] + ' ' + doc['wikipedia_title']
            else:
                doc_text = doc['text']
            documents.append(doc_text)
        doc_sim.similarity(documents)
        
    # print(doc_sim.saved_similarities)
    doc_sim.saved_similarities = [sim.reshape(-1, sim.shape[1], sim.shape[1]) for sim in doc_sim.saved_similarities]
    similarities = np.concatenate(doc_sim.saved_similarities, axis=0)
    print(similarities.shape)
    
    for i in range(similarities.shape[0]):
        for j in range(similarities.shape[1]-1):
            for k in range(j+1, similarities.shape[1]):
                _sim = (similarities[i, k, j] + similarities[i, j, k]) / 2
                similarities[i, j, k] = _sim
                similarities[i, k, j] = _sim
    
    np.save(f'{output_dir}/{output_name}', similarities)
    doc_sim.clear_similarity()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action='store_true')
    parser.add_argument("--process", action='store_true')
    args = parser.parse_args()
    
    main(args)
    
    #  python gen_doc_similarity.py --compute --corpus sphere --retriever bm25
    # CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python reranking/gen_doc_similarity.py --compute --retriever bm25