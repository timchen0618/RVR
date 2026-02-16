# import json
# import numpy as np
# from typing import List
# from tqdm import tqdm
# import regex
# import string
# import unicodedata


# def is_significantly_different(
#     scores_A: List,
#     scores_B: List,
#     alpha: float = 0.05,
#     n_trial: int = 10000,
# ) -> bool:
#     """Determine if two lists of model performance are significantly different."""
#     scores_A = np.array(scores_A)
#     scores_B = np.array(scores_B)
#     assert len(scores_A) == len(scores_B)

#     scores_A_mean = scores_A.mean()
#     scores_B_mean = scores_B.mean()
#     delta = scores_B_mean - scores_A_mean

#     count = 0
#     n_boostrap = len(scores_A)
#     for _ in range(n_trial):
#         rand_ids = np.random.choice(len(scores_A), size=n_boostrap, replace=True)
#         bootstrapped_scores_A = scores_A[rand_ids]
#         bootstrapped_scores_B = scores_B[rand_ids]

#         if delta > 0:
#             count += bootstrapped_scores_B.mean() > bootstrapped_scores_A.mean()
#         else:
#             count += bootstrapped_scores_B.mean() < bootstrapped_scores_A.mean()

#     p = 1 - count / n_trial
#     is_sig_diff = p <= alpha

#     print(f"\nModel A MRecall: {scores_A_mean:.4f}")
#     print(f"Model B MRecall: {scores_B_mean:.4f}")
#     print(f"Delta (B - A): {delta:.4f}")
#     print(f"p-value: {p:.4f} (threshold = {alpha})")
#     print(f"Result: {'SIGNIFICANT' if is_sig_diff else 'NOT SIGNIFICANT'}")

#     return is_sig_diff


# def _normalize(text):
#     return unicodedata.normalize('NFD', text)


# def normalize_answer(s):
#     def remove_articles(text):
#         return regex.sub(r'\b(a|an|the)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def has_answer(answers, text, tokenizer) -> bool:
#     """Check if a document contains an answer string."""
#     text = _normalize(text)
#     text = tokenizer.tokenize(text, uncased=True)

#     for answer in answers:
#         answer = _normalize(answer)
#         answer = tokenizer.tokenize(answer, uncased=True)
#         for i in range(0, len(text) - len(answer) + 1):
#             if answer == text[i: i + len(answer)]:
#                 return True
#     return False


# class SimpleTokenizer(object):
#     ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
#     NON_WS = r'[^\p{Z}\p{C}]'

#     def __init__(self):
#         self._regexp = regex.compile(
#             '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
#             flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
#         )

#     def tokenize(self, text, uncased=False):
#         matches = [m for m in self._regexp.finditer(text)]
#         if uncased:
#             tokens = [m.group().lower() for m in matches]
#         else:
#             tokens = [m.group() for m in matches]
#         return tokens


# def read_jsonl(file_path):
#     with open(file_path, 'r') as file:
#         return [json.loads(line) for line in file]


# def compute_mrecall_per_query(retrieved_docs_path, data_path, topk=100):
#     """Compute binary MRecall score for each query."""
#     dataset = read_jsonl(data_path)
#     tok = SimpleTokenizer()
#     retrieved_docs = read_jsonl(retrieved_docs_path)
    
#     assert len(dataset) == len(retrieved_docs)
    
#     mrecall_scores = []
    
#     for gold_inst, docs in tqdm(zip(dataset, retrieved_docs), total=len(dataset)):
#         pred_inst = []
        
#         if 'answer_list' in gold_inst:
#             answer_list_string = 'answer_list'
#         else:
#             answer_list_string = 'answers'
        
#         # Build pred_inst: list of lists, one per perspective/answer
#         for answer in gold_inst[answer_list_string]:
#             pred_inst.append([])
#             for doc in docs['ctxs'][:topk]:
#                 if answer_list_string == 'answer_list':
#                     pred = has_answer(answer['aliases'], doc['text'], tok)
#                 else:
#                     pred = has_answer(answer, doc['text'], tok)
#                 pred_inst[-1].append(pred)
        
#         # Skip empty instances
#         if len(pred_inst) == 0:
#             continue
            
#         # Compute MRecall for this query (binary: 1 or 0)
#         # topk_retrieved = number of documents actually retrieved
#         topk_retrieved = len(pred_inst[0]) if len(pred_inst) > 0 else 0
#         num_perspectives = len(pred_inst)
        
#         # Count how many perspectives are covered (at least one doc contains the answer)
#         num_covered = sum([any(preds_per_perspective) for preds_per_perspective in pred_inst])
        
#         if num_perspectives > topk_retrieved:
#             # If more perspectives than documents, need at least topk_retrieved perspectives covered
#             mrecall = int(num_covered >= topk_retrieved)
#         else:
#             # If perspectives <= documents, need ALL perspectives covered
#             mrecall = int(all([any(preds_per_perspective) for preds_per_perspective in pred_inst]))
        
#         mrecall_scores.append(mrecall)
    
#     return mrecall_scores


# def main():
#     # Paths to your model results
#     model_A_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/base/round2_merged.json'
#     model_B_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/both_obj/v100/infly/round2_merged.json'
#     gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    
#     topk = 100
    
#     print("Computing MRecall scores for Model A...")
#     mrecall_A = compute_mrecall_per_query(model_A_path, gt_data_path, topk=topk)
#     print("Computing MRecall scores for Model B...")
#     mrecall_B = compute_mrecall_per_query(model_B_path, gt_data_path, topk=topk)
   
    
#     print(f"\n{'='*50}")
#     print("BOOTSTRAP SIGNIFICANCE TEST")
#     print(f"{'='*50}")
    
#     is_sig = is_significantly_different(mrecall_A, mrecall_B, alpha=0.05, n_trial=10000)
#     print(f"isSig: {is_sig}")

#     model_A_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/base/round2_merged.json'
#     model_B_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/both_obj/istr/qwen3/round2_merged.json'
#     gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    
#     topk = 100
    
#     print("Computing MRecall scores for Model A...")
#     mrecall_A = compute_mrecall_per_query(model_A_path, gt_data_path, topk=topk)
#     print("Computing MRecall scores for Model B...")
#     mrecall_B = compute_mrecall_per_query(model_B_path, gt_data_path, topk=topk)
   
    
#     print(f"\n{'='*50}")
#     print("BOOTSTRAP SIGNIFICANCE TEST")
#     print(f"{'='*50}")
    
#     is_sig = is_significantly_different(mrecall_A, mrecall_B, alpha=0.05, n_trial=10000)
#     print(f"isSig: {is_sig}")

#     model_A_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/base/round2_merged.json'
#     model_B_path = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/both_obj/v100/contriever/round2_merged.json'
#     gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    
#     topk = 100
    
#     print("Computing MRecall scores for Model A...")
#     mrecall_A = compute_mrecall_per_query(model_A_path, gt_data_path, topk=topk)
#     print("Computing MRecall scores for Model B...")
#     mrecall_B = compute_mrecall_per_query(model_B_path, gt_data_path, topk=topk)
   
    
#     print(f"\n{'='*50}")
#     print("BOOTSTRAP SIGNIFICANCE TEST")
#     print(f"{'='*50}")
    
#     is_sig = is_significantly_different(mrecall_A, mrecall_B, alpha=0.05, n_trial=10000)
#     print(f"isSig: {is_sig}")
    
    

# if __name__ == "__main__":
#     main()
import json

def convert_webqsp_to_qampari_format(input_file, output_file):
    """
    Convert WebQSP format to QAMPARI format.
    
    Args:
        input_file: Path to WebQSP JSONL file
        output_file: Path to output JSON file
    """
    converted_data = []
    
    # Read WebQSP file (JSONL format)
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            webqsp_item = json.loads(line)
            
            # Create QAMPARI format entry
            qampari_item = {
                "question": webqsp_item["question_text"],
                "input": webqsp_item["question_text"],
                "id": idx,
                "answers": [""],
                "ctxs": [],
                "input_negative_ctxs": []
            }
            
            converted_data.append(qampari_item)
    
    # Write as JSON array (not JSONL)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"Converted {len(converted_data)} questions")
    print(f"Output saved to: {output_file}")

# Usage
if __name__ == "__main__":
    input_file = "/scratch/dq2024/diverse_retriever/data/dev_data_gt_webqsp.jsonl"
    output_file = "/scratch/dq2024/diverse_retriever/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_webqsp_concat_query_inp0.json"
    
    convert_webqsp_to_qampari_format(input_file, output_file)