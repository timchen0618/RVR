import json
import numpy as np
from typing import List
from tqdm import tqdm
import regex
import string
import unicodedata


def is_significantly_different(
    scores_A: List,
    scores_B: List,
    alpha: float = 0.05,
    n_trial: int = 10000,
    metric_name: str = "Score"
) -> bool:
    """Determine if two lists of model performance are significantly different."""
    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)
    assert len(scores_A) == len(scores_B)

    scores_A_mean = scores_A.mean()
    scores_B_mean = scores_B.mean()
    delta = scores_B_mean - scores_A_mean

    count = 0
    n_boostrap = len(scores_A)
    for _ in range(n_trial):
        rand_ids = np.random.choice(len(scores_A), size=n_boostrap, replace=True)
        bootstrapped_scores_A = scores_A[rand_ids]
        bootstrapped_scores_B = scores_B[rand_ids]

        if delta > 0:
            count += bootstrapped_scores_B.mean() > bootstrapped_scores_A.mean()
        else:
            count += bootstrapped_scores_B.mean() < bootstrapped_scores_A.mean()

    p = 1 - count / n_trial
    is_sig_diff = p <= alpha

    print(f"\nModel A {metric_name}: {scores_A_mean:.4f}")
    print(f"Model B {metric_name}: {scores_B_mean:.4f}")
    print(f"Delta (B - A): {delta:.4f}")
    print(f"p-value: {p:.4f} (threshold = {alpha})")
    print(f"Result: {'SIGNIFICANT' if is_sig_diff else 'NOT SIGNIFICANT'}")

    return is_sig_diff


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
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def compute_mrecall_per_query(retrieved_docs_path, data_path, topk=100):
    """Compute binary MRecall score for each query."""
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    
    assert len(dataset) == len(retrieved_docs)
    
    mrecall_scores = []
    
    for gold_inst, docs in tqdm(zip(dataset, retrieved_docs), total=len(dataset)):
        pred_inst = []
        
        if 'answer_list' in gold_inst:
            answer_list_string = 'answer_list'
        else:
            answer_list_string = 'answers'
        
        # Build pred_inst: list of lists, one per perspective/answer
        for answer in gold_inst[answer_list_string]:
            pred_inst.append([])
            for doc in docs['ctxs'][:topk]:
                if answer_list_string == 'answer_list':
                    pred = has_answer(answer['aliases'], doc['text'], tok)
                else:
                    pred = has_answer(answer, doc['text'], tok)
                pred_inst[-1].append(pred)
        
        # Skip empty instances
        if len(pred_inst) == 0:
            continue
            
        # Compute MRecall for this query (binary: 1 or 0)
        topk_retrieved = len(pred_inst[0]) if len(pred_inst) > 0 else 0
        num_perspectives = len(pred_inst)
        
        # Count how many perspectives are covered (at least one doc contains the answer)
        num_covered = sum([any(preds_per_perspective) for preds_per_perspective in pred_inst])
        
        if num_perspectives > topk_retrieved:
            # If more perspectives than documents, need at least topk_retrieved perspectives covered
            mrecall = int(num_covered >= topk_retrieved)
        else:
            # If perspectives <= documents, need ALL perspectives covered
            mrecall = int(all([any(preds_per_perspective) for preds_per_perspective in pred_inst]))
        
        mrecall_scores.append(mrecall)
    
    return mrecall_scores


def compute_recall_per_query(retrieved_docs_path, data_path, topk=100):
    """Compute Recall score for each query (proportion of answers covered)."""
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    
    assert len(dataset) == len(retrieved_docs)
    
    recall_scores = []
    
    for gold_inst, docs in tqdm(zip(dataset, retrieved_docs), total=len(dataset)):
        pred_inst = []
        
        if 'answer_list' in gold_inst:
            answer_list_string = 'answer_list'
        else:
            answer_list_string = 'answers'
        
        # Build pred_inst: list of lists, one per perspective/answer
        for answer in gold_inst[answer_list_string]:
            pred_inst.append([])
            for doc in docs['ctxs'][:topk]:
                if answer_list_string == 'answer_list':
                    pred = has_answer(answer['aliases'], doc['text'], tok)
                else:
                    pred = has_answer(answer, doc['text'], tok)
                pred_inst[-1].append(pred)
        
        # Skip empty instances
        if len(pred_inst) == 0:
            continue
        
        # Compute Recall for this query (proportion of perspectives covered)
        recall = sum([any(preds_per_perspective) for preds_per_perspective in pred_inst]) / len(pred_inst)
        recall_scores.append(recall)
    
    return recall_scores


def main():
   
    # Paths to your model results


    model_A_path = '/scratch/dq2024/diverse_retriever/inference/quest/contriever/quest_contriever_pretrained_t100.json'
    model_B_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/contriever/round2_merged.json'
    gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_quest.jsonl'
    topk = 100
    print("Computing MRecall scores for Model A...")
    mrecall_A = compute_mrecall_per_query(model_A_path, gt_data_path, topk=topk)
    print("Computing MRecall scores for Model B...")
    mrecall_B = compute_mrecall_per_query(model_B_path, gt_data_path, topk=topk)
    print("Computing Recall scores for Model A...")
    recall_A = compute_recall_per_query(model_A_path, gt_data_path, topk=topk)
    print("Computing Recall scores for Model B...")
    recall_B = compute_recall_per_query(model_B_path, gt_data_path, topk=topk)
    print(f"\n{'='*50}")
    print("BOOTSTRAP SIGNIFICANCE TEST - MRecall")
    print(f"{'='*50}")
    is_sig_mrecall = is_significantly_different(mrecall_A, mrecall_B, alpha=0.05, n_trial=10000, metric_name="MRecall")
    print(f"MRecall is Significant: {is_sig_mrecall}")
    print(f"\n{'='*50}")
    print("BOOTSTRAP SIGNIFICANCE TEST - Recall")
    print(f"{'='*50}")
    is_sig_recall = is_significantly_different(recall_A, recall_B, alpha=0.05, n_trial=10000, metric_name="Recall")
    print(f"Recall is Significant: {is_sig_recall}")


    model_A_path = '/scratch/dq2024/diverse_retriever/inference/quest/qwen3/quest_pretrained_qwen3_t100.json'
    model_B_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/qwen3/round2_merged.json'
    gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_quest.jsonl'
    topk = 100
    print("Computing MRecall scores for Model A...")
    mrecall_A = compute_mrecall_per_query(model_A_path, gt_data_path, topk=topk)
    print("Computing MRecall scores for Model B...")
    mrecall_B = compute_mrecall_per_query(model_B_path, gt_data_path, topk=topk)
    print("Computing Recall scores for Model A...")
    recall_A = compute_recall_per_query(model_A_path, gt_data_path, topk=topk)
    print("Computing Recall scores for Model B...")
    recall_B = compute_recall_per_query(model_B_path, gt_data_path, topk=topk)
    print(f"\n{'='*50}")
    print("BOOTSTRAP SIGNIFICANCE TEST - MRecall")
    print(f"{'='*50}")
    is_sig_mrecall = is_significantly_different(mrecall_A, mrecall_B, alpha=0.05, n_trial=10000, metric_name="MRecall")
    print(f"MRecall is Significant: {is_sig_mrecall}")
    print(f"\n{'='*50}")
    print("BOOTSTRAP SIGNIFICANCE TEST - Recall")
    print(f"{'='*50}")
    is_sig_recall = is_significantly_different(recall_A, recall_B, alpha=0.05, n_trial=10000, metric_name="Recall")
    print(f"Recall is Significant: {is_sig_recall}")

    model_A_path = '/scratch/dq2024/diverse_retriever/inference/quest/infly/quest_infly_pretrained_t100.json'
    model_B_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/infly/round2_merged.json'
    gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_quest.jsonl'
    topk = 100
    print("Computing MRecall scores for Model A...")
    mrecall_A = compute_mrecall_per_query(model_A_path, gt_data_path, topk=topk)
    print("Computing MRecall scores for Model B...")
    mrecall_B = compute_mrecall_per_query(model_B_path, gt_data_path, topk=topk)
    print("Computing Recall scores for Model A...")
    recall_A = compute_recall_per_query(model_A_path, gt_data_path, topk=topk)
    print("Computing Recall scores for Model B...")
    recall_B = compute_recall_per_query(model_B_path, gt_data_path, topk=topk)
    print(f"\n{'='*50}")
    print("BOOTSTRAP SIGNIFICANCE TEST - MRecall")
    print(f"{'='*50}")
    is_sig_mrecall = is_significantly_different(mrecall_A, mrecall_B, alpha=0.05, n_trial=10000, metric_name="MRecall")
    print(f"MRecall is Significant: {is_sig_mrecall}")
    print(f"\n{'='*50}")
    print("BOOTSTRAP SIGNIFICANCE TEST - Recall")
    print(f"{'='*50}")
    is_sig_recall = is_significantly_different(recall_A, recall_B, alpha=0.05, n_trial=10000, metric_name="Recall")
    print(f"Recall is Significant: {is_sig_recall}")

    

    
    
    
    
    

if __name__ == "__main__":
    main()