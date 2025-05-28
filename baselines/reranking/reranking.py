from utils.data_utils import collect_retrieval_results, write_jsonl, read_jsonl
import logging
import argparse
import json
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# def collect_retrieval_results(corpus, retriever)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')
            

class Reranker:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        self.args = args
        self.similarity_matrix = None
        self._lambda = args._lambda
        max_score_map = {"bm25":110.2, "nv":0.695, "contriever":2.164, "tart": 1.0}
        self.max_score = max_score_map[args.retriever]
        self.input_doc_score = 100
        # self.similarity_matrix = np.load(args.similarity_matrix)

    def rerank(self, retrieval_results, keep_top_k=3, compute_upto_top_k=False):
        logger.info(f"Reranking {len(retrieval_results)} instances while keeping {keep_top_k} top documents")
        assert 'ctxs' in retrieval_results[0]
        assert len(retrieval_results[0]['ctxs']) == self.similarity_matrix.shape[1], (len(retrieval_results[0]['ctxs']), self.similarity_matrix.shape[1])
        
        # compute max score
        self.max_scores = []
        for inst in retrieval_results:
            max_score = 0
            for j, doc in enumerate(inst['ctxs']):
                if j >= keep_top_k:
                    max_score = max(max_score, float(doc['score']))
            self.max_scores.append(max_score)
        # print('max_scores:', self.max_scores)
        
        data_idx = 0
        for inst in tqdm(retrieval_results):
            documents = inst['ctxs']
            
            ## keep top k documents (if there are k input documents, keep them at the top)
            if keep_top_k > 0:
                new_document_ids_n_scores = []
                for i in range(keep_top_k):
                    new_document_ids_n_scores.append((i, self.input_doc_score))
                if compute_upto_top_k:
                    input_doc_ids = [i for i, _ in new_document_ids_n_scores]
            else:
                new_document_ids_n_scores = [(0, float(documents[0]['score']))]
            
            for _ in range(len(documents) - keep_top_k):
                if keep_top_k > 0 and compute_upto_top_k:
                    new_document_id, mmr_score = self.add_one_document(documents, new_document_ids_n_scores, data_idx, input_doc_ids=input_doc_ids)
                else:
                    new_document_id, mmr_score = self.add_one_document(documents, new_document_ids_n_scores, data_idx)
                new_document_ids_n_scores.append((new_document_id, mmr_score))
                
            inst['ctxs'] = []
            
            for id_n_score in new_document_ids_n_scores:
                doc = documents[id_n_score[0]]
                doc['score'] = id_n_score[1]
                inst['ctxs'].append(doc)
            
            if keep_top_k > 0:
                inst['ctxs'] = inst['ctxs'][keep_top_k:]
            data_idx += 1

        return retrieval_results
    
    
    def rerank_average_input(self, retrieval_results, keep_top_k=3):
        logger.info(f"Reranking {len(retrieval_results)} instances while keeping {keep_top_k} top documents")
        assert 'ctxs' in retrieval_results[0]
        assert len(retrieval_results[0]['ctxs']) == self.similarity_matrix.shape[1], (len(retrieval_results[0]['ctxs']), self.similarity_matrix.shape[1])
        
        self.model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
        self.model.eval()

        # compute max score
        self.max_scores = []
        for inst in retrieval_results:
            max_score = 0
            for j, doc in enumerate(inst['ctxs']):
                if j >= keep_top_k:
                    max_score = max(max_score, float(doc['score']))
            self.max_scores.append(max_score)
        # print('max_scores:', self.max_scores)
        
        data_idx = 0
        for inst in tqdm(retrieval_results):
            documents = inst['ctxs']
            
            assert keep_top_k > 0
            # compute average document embeddings
            query_embeddings = self.model.encode([c['title'] + ' ' + c['text'] for c in inst['ctxs']], prompt_name="s2s_query")
            # print(query_embeddings.shape)
            doc_embeddings = self.model.encode([c['title'] + ' ' + c['text'] for c in inst['ctxs'][:keep_top_k]]).mean(axis=0).reshape(1, -1) 
            # print('size', query_embeddings.shape)
            # print('size', doc_embeddings.shape)
            
            similarities = self.model.similarity(query_embeddings, doc_embeddings) # (1, len(docs))
            # print(similarities.shape)
            
            new_document_ids_n_scores = []
            for i in range(keep_top_k):
                new_document_ids_n_scores.append((i, self.input_doc_score))
            
            input_doc_ids = [i for i, _ in new_document_ids_n_scores]
            
            TOP = 1000
            for i in range(len(inst['ctxs'])):
                if i < keep_top_k:
                    inst['ctxs'][i]['score'] = TOP - i
                new_score_i = float(inst['ctxs'][i]['score']) / self.max_score
                new_score_i -= self._lambda * similarities[i].item()
                inst['ctxs'][i]['score'] = new_score_i
            
            inst['ctxs'] = sorted(inst['ctxs'], key=lambda x: x['score'], reverse=True)
                
            # inst['ctxs'] = []
            
            # for id_n_score in new_document_ids_n_scores:
            #     doc = documents[id_n_score[0]]
            #     doc['score'] = id_n_score[1]
            #     inst['ctxs'].append(doc)
            
            if keep_top_k > 0:
                inst['ctxs'] = inst['ctxs'][keep_top_k:]
            data_idx += 1

        return retrieval_results
    
    def rerank_cross_encoder(self, retrieval_results, model_name='bge'):
        logger.info(f"Reranking {len(retrieval_results)} instances")
        assert 'ctxs' in retrieval_results[0]
        
        if model_name == 'bge':
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        elif model_name == 'jina':
            from transformers import AutoModelForSequenceClassification

            reranker = AutoModelForSequenceClassification.from_pretrained(
                'jinaai/jina-reranker-v2-base-multilingual',
                torch_dtype="auto",
                trust_remote_code=True,
            )
            reranker.to('cuda') # or 'cpu' if no GPU is available
            reranker.eval()
        else:
            raise NotImplementedError

        for inst in tqdm(retrieval_results):
            query_passage_pairs = []
            for i in range(len(inst['ctxs'])):
                query_passage_pairs.append([inst['question'], inst['ctxs'][i]['title'] + ' ' + inst['ctxs'][i]['text']])
            
            if model_name == 'bge':
                scores = reranker.compute_score(query_passage_pairs, normalize=False)
            elif model_name == 'jina':
                scores = reranker.compute_score(query_passage_pairs, max_length=1024)
            else:
                raise NotImplementedError
            print(scores)
            assert len(scores) == len(inst['ctxs'])
            for i in range(len(inst['ctxs'])):
                inst['ctxs'][i]['score'] = scores[i]
            
            inst['ctxs'] = sorted(inst['ctxs'], key=lambda x: x['score'], reverse=True)
            
        return retrieval_results
    
    def load_similarity_matrix(self, similarity_matrix_path):
        self.similarity_matrix = np.load(similarity_matrix_path)

    def add_one_document(self, documents, retrieved_document_ids_n_scores, data_idx, input_doc_ids=[]):
        mmr = -100
        new_document_id = 0
        retrieved_document_ids = [i for i, _ in retrieved_document_ids_n_scores]
        for i in range(len(documents)):
            if i not in retrieved_document_ids:
                mmr_i = self._lambda * float(documents[i]['score']) / self.max_score
                # max_score = self.max_scores[data_idx] if self.max_scores[data_idx] != 0 else self.max_score
                # mmr_i = self._lambda * float(documents[i]['score']) / max_score
                
                # compute maximum similarity 
                max_sim_between_docs = -100
                if len(input_doc_ids) > 0:
                    doc_ids_to_compare = input_doc_ids
                else:
                    doc_ids_to_compare = retrieved_document_ids
                for j in doc_ids_to_compare:
                    # sim_i_j = self.similarity(i, j)
                    assert i != j
                    sim_i_j = self.similarity_matrix[data_idx][i][j]
                    if sim_i_j > max_sim_between_docs:
                        max_sim_between_docs = sim_i_j
                 
                mmr_i -= (1 - self._lambda) * max_sim_between_docs
        
                if mmr_i > mmr:
                    mmr = mmr_i
                    new_document_id = i
        return new_document_id, mmr


    def similarity(self, i, j):
        assert i != j
        return self.similarity_matrix[i, j]
        # return 0.1
        def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'
        
        # Each query must come with a one-sentence instruction that describes the task
        task = 'Given a document, retrieve similar documents.'
        queries = [
            get_detailed_instruct(task, document1)
        ]
        # No need to add instruction for retrieval documents
        documents = [document2]
        input_texts = queries + documents


        max_length = 4096
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        print(scores.tolist())
        
        return scores[0]
    
    
def compute_max_score(retrieval_results):
    max_score = 0
    avg_scores = []
    for k, results in enumerate(retrieval_results):
        for i, inst in enumerate(results):
            for j, doc in enumerate(inst['ctxs']):
                assert 'score' in doc, (doc, i, j, k)
                max_score = max(max_score, float(doc['score']))
                avg_scores.append(float(doc['score']))
    return max_score, np.mean(np.array(avg_scores))




    


def main(args):
    # data_types = ['quest']
    logger.info('collecting retrieval results')
    rootdir = '/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/'
    
    output_dir = '/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/qampari_2nd_stage'
    
    prefix = 'qampari_dev_2nd_stage_qampari_corpus'
    input_file = f'/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_qampari/{prefix}_org_org_query.json'
    similarity_matrix_path = f'{output_dir}/{prefix}_org_similarity.npy'
    retrieval_results = read_jsonl(input_file)
    raw_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus.jsonl')
    assert len(retrieval_results) == len(raw_data), (len(retrieval_results), len(raw_data))
    
    # # compute max score
    # max_score, mean_score = compute_max_score([retrieval_results])
    # print('max_score:', max_score)
    # print('mean_score:', mean_score)
    # exit(0)
    
    logger.info(f"Reranking results for {input_file}")
    if not args.cross_encoder:
        for ret_inst, inst in tqdm(zip(retrieval_results, raw_data)):
            assert (ret_inst['question'] == inst['question'] or ret_inst['question'] == inst['org_q']) 
            assert len(inst['input_negative_ctxs']) == 3
            ret_inst['ctxs'] = inst['input_negative_ctxs'] + ret_inst['ctxs']
        
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = None, None
    
    reranker = Reranker(tokenizer=tokenizer, model=model, device=device, args=args)

    reranker.load_similarity_matrix(similarity_matrix_path)
    if args.average_input:
        average_input_str = "_avg_input"
        reranked_results = reranker.rerank_average_input(retrieval_results, keep_top_k=3)
    elif args.cross_encoder:
        average_input_str = "_cross_encoder"
        reranked_results = reranker.rerank_cross_encoder(retrieval_results, model_name=args.reranker)
    else:
        average_input_str = ""
        reranked_results = reranker.rerank(retrieval_results, keep_top_k=3, compute_upto_top_k=False)
    
    input_file_name = Path(input_file).stem
    
    if args.cross_encoder:
        write_jsonl(f'{output_dir}/{input_file_name}_reranked{average_input_str}_{args.reranker}_org_query.jsonl', reranked_results)
        logger.info(f"Reranked results saved to {output_dir}/{input_file_name}_reranked{average_input_str}_{args.reranker}_org_query.jsonl")
    else:
        write_jsonl(f'{output_dir}/{input_file_name}_reranked{average_input_str}_l{args._lambda}.jsonl', reranked_results)
        logger.info(f"Reranked results saved to {output_dir}/{input_file_name}_reranked{average_input_str}_l{args._lambda}.jsonl")
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='contriever')
    parser.add_argument("--_lambda", type=float, default=0.5)
    parser.add_argument("--reranker", type=str, default='')
    parser.add_argument("--average_input", action='store_true')
    parser.add_argument("--cross_encoder", action='store_true')
    args = parser.parse_args()
    
    main(args)

    # max score 
    # wiki bm25 max_score: 63.47243881225586     mean_score: 12.411771381240964
    # wiki dpr max_score: 100.568115             mean score: 77.29109310180797 
    # wiki contriever max_score: 2.1931763       mean_score: 1.410406476006172 
    # sphere bm25 max_score: 99.82140350341797   mean_score: 15.574951505362245
    # sphere dpr max_score: 243.86131            mean_score: 201.92912051533668
    # sphere contriever max_score: 2.5431857     mean_score: 1.6246204772375314



