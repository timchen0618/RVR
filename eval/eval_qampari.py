import json
from sklearn.metrics import precision_score, recall_score, f1_score
import regex
import string
from tqdm import tqdm
import unicodedata
import pytrec_eval
from nltk import word_tokenize

def score_recall(preds):
    # average 
    recalls = []
    for inst in preds:
        if len(inst) == 0:
            print("skipping inst",flush=True)
            continue

        recalls.append(sum([any(preds_per_perspective) for preds_per_perspective in inst])/len(inst))
    print(len(recalls), flush=True)

    return sum(recalls) / float(len(recalls))

def score_mrecall(preds):
    # average 
    mrecalls = []
    topk = len(preds[0][0])
    for inst in preds:  # compute mrecall for each instance
        if len(inst) > topk:
            mrecalls.append(int(sum([any(preds_per_perspective) for preds_per_perspective in inst])>=topk))
        else:
            mrecalls.append(int(all([any(preds_per_perspective) for preds_per_perspective in inst])))

    return sum(mrecalls) / float(len(mrecalls))

def score_precision(preds, topk):
    # average 
    precisions = []
    for inst in preds:
        assert len(inst[0]) >= topk, len(inst[0])
        # if len(inst[0]) < topk:
        #     topk = len(inst[0])
        num_perspective_containing_docs = 0
        for j in range(topk):
            contain_any_perspective = False
            for p in inst:
                if p[j]:
                    contain_any_perspective = True
                    break
            if contain_any_perspective:
                num_perspective_containing_docs += 1

        precisions.append(num_perspective_containing_docs / topk)

    return sum(precisions) / float(len(precisions))

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def read_jsonl(file_path):
    import json
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
    
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def process_answer(answer_str):
    if isinstance(answer_str, list):
        return [normalize_answer(ans) for ans in answer_str]
    answer_list = []
    for ans in answer_str.split('\n'):
        ans = ' '.join(ans.split(' ')[1:])
        # answer_list.append(normalize_answer(ans.strip()[3:].strip()))
        answer_list.append(normalize_answer(ans))
    return answer_list

def process_reference(inst):
    # return a list
    answer_list = []
    for answer in inst['answer_list']:
        answer_list.append(answer['aliases'])
    answer_list = [item for sublist in answer_list for item in sublist]
    answer_list = [normalize_answer(ans) for ans in answer_list]
    return answer_list


def evaluate(predictions, references):
    predictions = set(list(predictions))
    references = set(list(references))
    num_intersections = 0
    for pred in predictions:
        for ref in references:
            if pred == ref:
                num_intersections += 1
                break
    if len(predictions) == 0:
        return 0, 0, 0
    precision = num_intersections / len(predictions)
    recall = num_intersections / len(references)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1




def eval_retrieve_docs(retrieved_docs_path, data_path, has_gold_id=False, topk=100, selected_indices=None):
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    len_docs = []
    # precisions = []
    preds = []
    qrel = {}
    run = {}
    mrrs = []

    
    assert len(dataset) == len(retrieved_docs), f'Length of dataset and retrieved_docs do not match: {len(dataset)} vs {len(retrieved_docs)}'
    if selected_indices is not None:
        dataset = [dataset[i] for i in selected_indices]
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    qid = 0
    for gold_inst, docs in tqdm(zip(dataset, retrieved_docs)):
        pred_inst = []
        mrr_inst = {}
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst['org_q']
        docs_question = docs['question_text'] if 'question_text' in docs else docs['question']
        
        
        #assert gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
        # valid_answers = process_reference(gold_inst)
        # print(valid_answers)
        
        # for ans in valid_answers:
        if has_gold_id:
            qrel[str(qid)] = {}
            if 'positive_ctxs' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['positive_ctxs']]
            elif 'ground_truths' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['ground_truths']]
            else:
                raise ValueError('No positive_ctxs or ground_truths')
            for cluster in gold_clusters:
                gold_ids_per_cluster = [doc['id'] for doc in cluster]
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for rank, doc in enumerate(docs['ctxs'][:topk]):
                    pred = (doc['id'] in set(gold_ids_per_cluster))
                    pred_inst[-1].append(pred)
                    
                    # handle MRR if prediction correct
                    for gold_id in gold_ids_per_cluster:
                        if doc['id'] == gold_id:
                            mrr_inst[gold_id] = 1 / (rank + 1)
                    
                for gold_id in gold_ids_per_cluster:
                    qrel[str(qid)][gold_id] = 1
                    if gold_id not in mrr_inst: # if not found
                        mrr_inst[gold_id] = 0
                        
            run[str(qid)] = {}
            for rank, doc in enumerate(docs['ctxs'][:topk]):
                # run[str(qid)][doc['id']] = 1 / (rank + 1)
                run[str(qid)][doc['id']] = float(doc['score'])
        else:
            
                
            if 'answer_list' in gold_inst:
                answer_list_string = 'answer_list'
            
            else:
                answer_list_string = 'answers'
            
            for answer_idx, answer in enumerate(gold_inst[answer_list_string]):
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for rank, doc in enumerate(docs['ctxs'][:topk]):
                    if answer_list_string == 'answer_list':
                        #print(answer)
                        pred = has_answer(answer['aliases'], doc['text'], tok)
                       # print(pred)
                    else:
                        pred = has_answer(answer, doc['text'], tok)
                    pred_inst[-1].append(pred)

                    # NOTE: I added this MRR logic
                    if pred:
                        answer_key = f"answer_{answer_idx}"
                        if answer_key not in mrr_inst:  # Only record first occurrence
                            mrr_inst[answer_key] = 1 / (rank + 1)

            for answer_idx in range(len(gold_inst[answer_list_string])):
                answer_key = f"answer_{answer_idx}"
                if answer_key not in mrr_inst:
                    mrr_inst[answer_key] = 0

        
        preds.append(pred_inst)  
        qid += 1  
        # compute average MRR for this instance
        _mrr = 0
        for _, value in mrr_inst.items():
            _mrr += value
        if len(mrr_inst) > 0:
            _mrr /= len(mrr_inst)
        mrrs.append(_mrr)
        
        # precisions.append(sum([any(doc[pred_str]) for doc in docs[:topk]]) / topk)

        original_count = len(preds)
        filtered_preds = []
        filtered_mrrs = []
        
        for i, inst in enumerate(preds):
            if len(inst) > 0:
                filtered_preds.append(inst)
                filtered_mrrs.append(mrrs[i])
        
        preds = filtered_preds
        mrrs = filtered_mrrs
        
        if len(preds) != original_count:
            print(f"Filtered out {original_count - len(preds)} questions with no answers", flush=True)

    if has_gold_id:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
        scores_dict = evaluator.evaluate(run)
        ndcgs = []
        maps = []
        for qid in scores_dict:
            ndcgs.append(scores_dict[qid]['ndcg'])
            maps.append(scores_dict[qid]['map'])
        nDCG = sum(ndcgs) / len(ndcgs)
        mAP = sum(maps) / len(maps)
        
    MRR = sum(mrrs) / len(mrrs)
    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    precision_score = score_precision(preds, topk)
    if has_gold_id:
        print(f'Evidence Recall - MRecall@{topk}: {100*mrecall_score:.2f} | Recall@{topk}: {100*recall_score:.2f} | Precision@{topk}: {100*precision_score:.2f} | mAP@{topk}: {mAP:.4f} | nDCG@{topk}: {nDCG:.4f} | MRR@{topk}: {MRR:.4f}')
        return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score), '%2.4f'%(mAP), '%2.4f'%(nDCG), '%2.4f'%(MRR), qrel, run
    else:
        print(f'Answer Recall - MRecall@{topk}: {100*mrecall_score:.4f} | Recall@{topk}: {100*recall_score:.4f} | Precision@{topk}: {100*precision_score:.4f} | MRR@{topk}: {MRR:.4f}', flush=True)
        return '%2.4f'%(100*mrecall_score), '%2.4f'%(100*recall_score), '%2.4f'%(100*precision_score), '%2.4f'%(MRR), qrel, run
    # print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | nDCG: {nDCG:.4f} | mAP: {mAP:.4f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    

# # If whether repeats flag is set to true
# def eval_retrieve_docs_for_repeats(retrieved_docs_path, data_path, topk=100):
#     dataset = read_jsonl(data_path)
#     tok = SimpleTokenizer()
#     retrieved_docs = read_jsonl(retrieved_docs_path)
#     len_docs = []
#     preds = []
    
#     for gold_inst, docs in zip(dataset, retrieved_docs):
#         pred_inst = []
#         # valid_answers = process_reference(gold_inst)
#         # print(valid_answers)
#         # for ans in valid_answers:
#         gold_indices = [doc['id'] for doc in gold_inst['input_negative_ctxs']]

#         for gold_index in gold_indices:
#             pred_inst.append([])
#             len_docs.append(len(docs['ctxs']))
#             for doc in docs['ctxs'][:topk]:
#                 # print(gold_index, doc['id'])
#                 pred = gold_index == doc['id']
#                 pred_inst[-1].append(pred)
        
#         preds.append(pred_inst)    

#     mrecall_score = score_mrecall(preds)
#     recall_score = score_recall(preds)
#     precision_score = score_precision(preds, topk)
#     print(f'Whether Repeats - MRecall@{topk}: {100*mrecall_score:.2f} | Recall@{topk}: {100*recall_score:.2f} | Precision@{topk}: {100*precision_score:.2f}')
#     print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
#     return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score)


def evaluate(input_docs, evaluate_repeat, has_gold_id, file_list, gt_data_path):
    for num_input in [input_docs]:  # Number of input documents: 0, 1, 2, 3
            
        #file_list = ['/scratch/dq2024/diverse_retriever/training/contriever/contriever_msmarco_qampari/qampari_corpus_finetuned_gru_only_random.json']          # files to be evaluated
        
        selected_indices = None   # To select a subset of the data. None -> use all the data. 
        
        for _file in file_list:
            if 'v100r@50' in _file:
                k = 50
            elif 'v50r@25' in _file:
                k = 25
            else:
                k = 100
            scores_per_file = []
            mrecalls = []
            recalls = []
            for TOPK in [k]:
                print(f"\nResults for {num_input}")
                scores_per_file += [0]
                print('--'*20)
                print(_file)
                scores = eval_retrieve_docs(
                    _file,
                    gt_data_path,
                    has_gold_id=has_gold_id,
                    topk=TOPK,
                    selected_indices=selected_indices
                )
                qrels = scores[-2]
                runs = scores[-1]

                # DEBUG
                # first_qid     = list(qrels.keys())[0]
                # gt_ids        = set(qrels[first_qid].keys())
                # retrieved_ids = set(runs[first_qid].keys())
                # print(f"Debug qid={first_qid!r}")
                # print("  gold ids:     ", gt_ids)
                # print("  retrieved ids:", retrieved_ids)
                # print("  intersection: ", gt_ids & retrieved_ids)
                
                scores = scores[:-2]
                if evaluate_repeat:
                    repeat_scores = eval_retrieve_docs_for_repeats(
                        _file,
                        gt_data_path,
                        topk=TOPK
                    )
                    scores = list(scores) + list(repeat_scores)
                else:
                    scores = list(scores)
                scores_per_file += scores

                mrecalls.append(scores[0])
                recalls.append(scores[1])


def write_jsonl(data, file_path):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


    
def main():

    
    evaluate_repeat = False   # whether to evaluate if the input documents are retrieved (repeated)
    has_gold_id = False        # whether the gold ids are provided, False = AR
    input_docs = 0
   
   
    file_list = [
        # '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t200_promote-qwen3_0-3_retrieved.json',
        #  '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t200_promote-qwen3_0-6_retrieved.json',
        #   '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t200_promote-qwen3_0-12_retrieved.json',
        #    '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-3_retrieved.json',
        #     '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-6_retrieved.json',
        #      '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-12_retrieved.json',
           
       #     '',
       #     '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/qampari_base_finetuned_infly_0-3_q3verif_input_test.json',
    #   '/scratch/dq2024/diverse_retriever/eval/contriever/base/MERGED_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json'
        #   '/scratch/dq2024/diverse_retriever/eval/infly/merged_qampari_base_finetuned_infly_0-12_input_test.json',
       #    '/scratch/dq2024/diverse_retriever/eval/infly/merged_qampari_base_finetuned_infly_0-3_input_test.json'
        #    '/scratch/dq2024/diverse_retriever/eval/qwen3/base/MERGED_qampari_qwen3_finetuned_base_0-3_input_test.json',
        #    '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-6_test.json',
        #    '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-12_test.json'
           #  '/scratch/dq2024/diverse_retriever/eval/contriever/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-3_retrieved.json',
            #  '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t200_promote-qwen3_0-12_retrieved.json',
            #  '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-3_retrieved.json',
            #  '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-6_retrieved.json',
            #  '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_finetuned_r1q3verif_std_t500_promote-qwen3_0-12_retrieved.json',


       # '/scratch/dq2024/diverse_retriever/eval/qwen3/base/MERGED_qampari_contreiver_tuned_base_model_0-3_qwen3-verif_input_test.json'
       # '/scratch/dq2024/diverse_retriever/eval/qwen3/base/MERGED_qampari_qwen3_finetuned_base_0-3_input_test.json',
        #'/scratch/dq2024/diverse_retriever/query_rewrite/infly/qampari_infly_finetuned_base_merged_query1_query2.json'
       # '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_std_t200_promote-qwen3_0-3_retrieved.json',
        #'/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_std_t200_promote-qwen3_0-6_retrieved.json',
        #'/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_std_t200_promote-qwen3_0-12_retrieved.json',
        #'/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_std_t500_promote-qwen3_0-3_retrieved.json',
        #'/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_std_t500_promote-qwen3_0-6_retrieved.json',
       # '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/MERGED_qampari_finetuned_std_t500_promote-qwen3_0-12_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0_200docs.json',
   # '/scratch/dq2024/diverse_retriever/eval/infly/standard/MERGED_ALL_GOLD_r1inf_r2std_AR_0-15_200docs.json'
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-1_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-2_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-3_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-6_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-9_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-12_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-15_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-18_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-21_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-24_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-27_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-30_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-35_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-40_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-45_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-50_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-55_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-60_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-65_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-70_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-75_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-80_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-85_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/max6_model/MERGED_ALL_GOLD_r1inf_r2std0-6_AR_0-90_200docs.json',

    #'/scratch/dq2024/diverse_retriever/eval/qwen3/standard/max12_model/MERGED_ALL_GOLD_r1qw_r2std0-3_AR__0-95_200docs.json'
    # '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t200_promote-qwen3_0-3_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t200_promote-qwen3_0-6_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t200_promote-qwen3_0-12_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t500_promote-qwen3_0-3_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t500_promote-qwen3_0-6_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t500_promote-qwen3_0-12_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/contriever/qwen3_verifier/MERGED_qampari_std_t200_promote-qwen3_0-3_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/contriever/qwen3_verifier/MERGED_qampari_std_t500_promote-qwen3_0-3_retrieved.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/MERGED_ALL_GOLD_r1inf_r2std_0-6_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/MERGED_ALL_GOLD_r1inf_r2std_0-12_200docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/MERGED_ALL_GOLD_r1inf_r2std_0-6_500docs.json',
    # '/scratch/dq2024/diverse_retriever/eval/infly/standard/MERGED_ALL_GOLD_r1inf_r2std_0-12_500docs.json'

    '/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/qampari_base_finetuned_infly_0-6_q3verif_input_test.json',
    '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_contreiver_tuned_base_model_0-6_r1q3-verif_input_test.json'


]
    file_list = ['/scratch/dq2024/diverse_retriever/eval/contriever/qwen3_verifier/MERGED_qampari_std_t200_promote-qwen3_0-3_retrieved_round3_100.json']
    # file_list = ['/scratch/dq2024/diverse_retriever/eval/query_rewrite/contriever/MERGED_contriever_query+keywords_max3_retrieved.json',
    #              '/scratch/dq2024/diverse_retriever/eval/query_rewrite/qwen3/MERGED_qwen3_query+keywords_max6_retrieved.json',
    #              '/scratch/dq2024/diverse_retriever/eval/query_rewrite/infly/MERGED_infly_query+keywords_max6_retrieved.json',
                 
    #              ]
    file_list = ['/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_base_only_random_v50_top25.json',
                 '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_corpus_base_only_random_v50_top25.json',
                 '/scratch/dq2024/diverse_retriever/eval/infly/base/qampari_corpus_base_only_random_v50_top25.json']
    
    file_list = ['/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3/merged_docs/round2_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3/merged_docs/round3_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3/merged_docs/round4_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3/merged_docs/round5_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v100r@50/qwen3/merged_docs/round2_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v100r@50/qwen3/merged_docs/round3_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v100r@50/qwen3/merged_docs/round4_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v100r@50/qwen3/merged_docs/round5_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/qwen3/merged_docs/round2_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/qwen3/merged_docs/round3_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/qwen3/merged_docs/round4_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/qwen3/merged_docs/round5_merged_100docs.json',]

    
    file_list = [#'/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/infly/merged_docs/round2_merged_100docs.json',
                 #'/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/infly/merged_docs/round3_merged_100docs.json',
               #  '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/infly/merged_docs/round4_merged_100docs.json',
               
                 
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/infly/merged_docs/round2_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/infly/merged_docs/round3_merged_100docs.json',
                 '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/v50r@25/infly/merged_docs/round4_merged_100docs.json',
                ]
    file_list = ['/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round2_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round2_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round3_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round3_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round4_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round4_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round5_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round5_merged.json'
                ]
    file_list = ['/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/base/round2_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/round2_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/base/round3_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/round3_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/base/round4_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/round4_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/base/round5_merged.json',
                '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/round5_merged.json'
                ]
    file_list = ['/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50_unique_inp/infly/base/round2_merged.json',
                 '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50_unique_inp/infly/round2_merged.json',
                 '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50_retr_unique_inp/infly/round2_merged.json',
                 '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50_retr+gt_unique_inp/infly/round2_merged.json',
                ]
    
    # file_list = ['/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json',
    #              '/scratch/dq2024/diverse_retriever/inference/qwen3/base/qampari_corpus_finetuned_base_t100.json',
    #              '/scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t100.json']
    file_list = [
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp1/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp1/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp2/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp2/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp3/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp3/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/infly/round2_merged.json',
        
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp12/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp12/infly/round2_merged.json',



        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp1/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp1/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp2/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp2/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp3/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp3/qwen3/round2_merged.json',

        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp12/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp12/qwen3/round2_merged.json',
    ]

    file_list = [
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100_2.json',
        '/scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_2_inp0_t100.json',
        '/scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t100.json',
        '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_base_only_random_inp0_t200.json',
        '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_base_finetuned_only_random_vanilla_inp0_t100.json',
        '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_base_finetuned_only_random_gold_inp0_AR_t100.json',
        '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_base_finetuned_only_random_gold_inp0_AR_t200.json',
       

        '/scratch/dq2024/diverse_retriever/inference/contriever/base/qampari_pretrained_contriever_t100.json',
        '/scratch/dq2024/diverse_retriever/inference/qwen3/base/qampari_pretrained_qwen3_t100.json',
        '/scratch/dq2024/diverse_retriever/inference/infly/base/qampari_pretrained_infly_t100.json'
    ]

    file_list = [
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/infly/round2_merged.json',
    ]

    file_list = [
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round3_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round4_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round5_merged.json',

        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round3_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round4_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round5_merged.json',

        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/base/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/base/round3_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/base/round4_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/base/round5_merged.json',

        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round3_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round4_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round5_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/wen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/base/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp1/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp2/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp3/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp6/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp9/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp0/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp1/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp2/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp3/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp6/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp9/qwen3/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp1/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp2/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp3/infly/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/infly/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp12/infly/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp0/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp1/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp2/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp3/qwen3/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp12/qwen3/round2_merged.json',

        


    ]



    # file_list = [
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/contriever/base/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v20/contriever/base/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v10/contriever/base/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/contriever/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v20/contriever/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v10/contriever/round2_merged.json',
    # ]


    # file_list = [
    #    # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/infly/base/round2_merged.json',
    #    # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v20/infly/base/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v10/infly/base/round2_merged.json',
    #    # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/infly/round2_merged.json',
    #     #'/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v20/infly/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v10/infly/round2_merged.json',

    #    # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/base/round2_merged.json',
    #     #'/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v20/qwen3/base/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v10/qwen3/base/round2_merged.json',
    #    # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/qwen3/round2_merged.json',
    #     #'/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v20/qwen3/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v10/qwen3/round2_merged.json',
    # ]

    # file_list = [
    #     #'/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp12/infly/round2_merged.json',
    #     '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp12/qwen3/round2_merged.json',
    # ]
    file_list = [
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/max12/inp12/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100itr/infly/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100itr/qwen3/round2_merged.json'

    ]
    file_list = [
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/contriever/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/contriever/base/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/contriever/pretrained/round2_merged.json',

        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/qwen3/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/qwen3/base/round2_merged.json',
        # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/qwen3/pretrained/round2_merged.json',

        '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/infly/round2_merged.json',
       # '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/infly/base/round2_merged.json',
      #  '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v100/infly/pretrained/round2_merged.json',
    ]
    
    file_list = [
       
           '/scratch/dq2024/ColBERT/outputs/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp0_retrieved.jsonl',
           '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/base/round2_merged.json',
           '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v/contriever/round2_merged.json',
    ]
    gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    has_gold_id = False        # whether the gold ids are provided, False = AR
    evaluate(input_docs, evaluate_repeat, has_gold_id, file_list, gt_data_path)

    file_list = [
        '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3_verifier/v50/contriever/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3_verifier/v50/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3_verifier/v50/infly/round2_merged.json',
        
        '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3_verifier/v50/contriever/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3_verifier/v50/qwen3/round2_merged.json',
        '/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/qwen3_verifier/v50/infly/round2_merged.json',
    ]

    gt_data_path = f'/scratch/dq2024/diverse_retriever/data/dev_data_gt_webqsp.jsonl'
    has_gold_id = False        # whether the gold ids are provided, False = AR
    evaluate(input_docs, evaluate_repeat, has_gold_id, file_list, gt_data_path)

                
if __name__ == "__main__":
    main()
