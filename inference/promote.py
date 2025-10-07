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
        recalls.append(sum([any(preds_per_perspective) for preds_per_perspective in inst])/len(inst))

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
    print(len(dataset))
    print(len(retrieved_docs))
    
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
        
        
        assert gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
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
            # NOTE: debug
            for x in gold_inst:
                print(f'{x} ---- {gold_inst[x]} \n')
                
            if 'answer_list' in gold_inst:
                answer_list_string = 'answer_list'
            
            else:
                answer_list_string = 'answers'
            
            for answer in gold_inst[answer_list_string]:
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for doc in docs['ctxs'][:topk]:
                    if answer_list_string == 'answer_list':
                        pred = has_answer(answer['aliases'], doc['text'], tok)
                    else:
                        pred = has_answer(answer, doc['text'], tok)
                    pred_inst[-1].append(pred)
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
        print(f'Answer Recall - MRecall@{topk}: {100*mrecall_score:.2f} | Recall@{topk}: {100*recall_score:.2f} | Precision@{topk}: {100*precision_score:.2f} | MRR@{topk}: {MRR:.4f}')
        return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score), '%2.4f'%(MRR), qrel, run
    # print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | nDCG: {nDCG:.4f} | mAP: {mAP:.4f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    

# If whether repeats flag is set to true
def eval_retrieve_docs_for_repeats(retrieved_docs_path, data_path, topk=100):
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    len_docs = []
    preds = []
    
    for gold_inst, docs in zip(dataset, retrieved_docs):
        pred_inst = []
        # valid_answers = process_reference(gold_inst)
        # print(valid_answers)
        # for ans in valid_answers:
        gold_indices = [doc['id'] for doc in gold_inst['input_negative_ctxs']]

        for gold_index in gold_indices:
            pred_inst.append([])
            len_docs.append(len(docs['ctxs']))
            for doc in docs['ctxs'][:topk]:
                # print(gold_index, doc['id'])
                pred = gold_index == doc['id']
                pred_inst[-1].append(pred)
        
        preds.append(pred_inst)    

    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    precision_score = score_precision(preds, topk)
    print(f'Whether Repeats - MRecall@{topk}: {100*mrecall_score:.2f} | Recall@{topk}: {100*recall_score:.2f} | Precision@{topk}: {100*precision_score:.2f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score)





def write_jsonl(data, file_path):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def extract_gold_documents(retrieved_docs_path, ground_truth_path, output_path, has_gold_id=False, topk=100, selected_indices=None):
    """Extract gold documents from retrieved documents and save to new file."""
    # Load data
    dataset = read_jsonl(ground_truth_path)
    retrieved_docs = read_jsonl(retrieved_docs_path)
    tok = SimpleTokenizer()
    
    # Validate data lengths
    assert len(dataset) == len(retrieved_docs), f'Length mismatch: {len(dataset)} vs {len(retrieved_docs)}'
    
    # Select subset if specified
    if selected_indices is not None:
        dataset = [dataset[i] for i in selected_indices]
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    gold_documents = []
    
    for idx, (gold_inst, docs) in enumerate(tqdm(zip(dataset, retrieved_docs), desc="Extracting gold documents")):
        # Validate questions match
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst['org_q']
        docs_question = docs['question_text'] if 'question_text' in docs else docs['question']
        
        assert gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
        
        question_gold_docs = {
            'question_id': idx,
            'question_text': gold_question,
            'gold_documents': []
        }
        
        if has_gold_id:
            # Use gold document IDs
            if 'positive_ctxs' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['positive_ctxs']]
            elif 'ground_truths' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['ground_truths']]
            else:
                raise ValueError('No positive_ctxs or ground_truths')
            
            # Collect all gold IDs
            all_gold_ids = set()
            for cluster in gold_clusters:
                gold_ids_per_cluster = [doc['id'] for doc in cluster]
                all_gold_ids.update(gold_ids_per_cluster)
            
            # Find retrieved documents that match gold IDs
            for rank, doc in enumerate(docs['ctxs'][:topk]):
                if doc['id'] in all_gold_ids:
                    gold_doc = {
                        'id': doc['id'],
                        'text': doc['text'],
                        'title': doc.get('title', ''),
                        'score': doc.get('score', 0),
                        'rank': rank + 1
                    }
                    question_gold_docs['gold_documents'].append(gold_doc)
        
        else:
            # Use answer matching
            answer_list_string = 'answer_list' if 'answer_list' in gold_inst else 'answers'
            
            for rank, doc in enumerate(docs['ctxs'][:topk]):
                contains_answer = False
                matching_answers = []
                
                for answer in gold_inst[answer_list_string]:
                    if answer_list_string == 'answer_list':
                        if has_answer(answer['aliases'], doc['text'], tok):
                            contains_answer = True
                            matching_answers.extend(answer['aliases'])
                    else:
                        if has_answer([answer], doc['text'], tok):
                            contains_answer = True
                            matching_answers.append(answer)
                
                if contains_answer:
                    gold_doc = {
                        'id': doc.get('id', f'doc_{rank}'),
                        'text': doc['text'],
                        'title': doc.get('title', ''),
                        'score': doc.get('score', 0),
                        'rank': rank + 1,
                        'matching_answers': list(set(matching_answers))
                    }
                    question_gold_docs['gold_documents'].append(gold_doc)
        
        # Only add if we found gold documents
        if question_gold_docs['gold_documents']:
            gold_documents.append(question_gold_docs)
    
    # Save extracted gold documents
    write_jsonl(gold_documents, output_path)

def rerank_with_gold_priority(retrieved_docs_path, ground_truth_path, output_path, has_gold_id=True, final_topk=100, initial_topk=200, selected_indices=None):
    """
    Rerank retrieved documents by putting gold documents first, then highest scoring non-gold docs.
    
    Args:
        retrieved_docs_path: Path to JSONL file with retrieved documents (top 200)
        ground_truth_path: Path to JSONL file with ground truth data
        output_path: Path to save the reranked documents (top 100)
        has_gold_id: Whether gold document IDs are provided
        final_topk: Final number of documents to keep (100)
        initial_topk: Number of documents to consider from retrieved (200)
        selected_indices: Optional subset of indices to process
    """
    # Load data
    dataset = read_jsonl(ground_truth_path)
    retrieved_docs = read_jsonl(retrieved_docs_path)
    tok = SimpleTokenizer()
    
    # Validate data lengths
    assert len(dataset) == len(retrieved_docs), f'Length mismatch: {len(dataset)} vs {len(retrieved_docs)}'
    
    # Select subset if specified
    if selected_indices is not None:
        dataset = [dataset[i] for i in selected_indices]
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    reranked_documents = []
    
    for idx, (gold_inst, docs) in enumerate(tqdm(zip(dataset, retrieved_docs), desc="Reranking documents")):
        # Validate questions match
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst['org_q']
        docs_question = docs['question_text'] if 'question_text' in docs else docs['question']
        
        assert gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
        
        # Create new document entry with same structure
        reranked_entry = {
            'question': docs.get('question', gold_question),
            'input': docs.get('input', gold_question), 
            'id': docs.get('id', idx),
            'answers': docs.get('answers', []),
            'ctxs': []
        }
        
        # Copy other fields if they exist
        if 'input_negative_ctxs' in docs:
            reranked_entry['input_negative_ctxs'] = docs['input_negative_ctxs']
        
        # Get initial retrieved documents (top 200)
        initial_docs = docs['ctxs'][:initial_topk]
        
        gold_docs = []
        non_gold_docs = []
        
        if has_gold_id:
            # Collect all gold IDs
            all_gold_ids = set()
            if 'positive_ctxs' in gold_inst:
                gold_clusters = gold_inst['positive_ctxs']
            elif 'ground_truths' in gold_inst:
                gold_clusters = gold_inst['ground_truths']
            else:
                raise ValueError('No positive_ctxs or ground_truths')
            
            for cluster in gold_clusters:
                for doc in cluster:
                    all_gold_ids.add(doc['id'])
            
            # Separate gold and non-gold documents
            for doc in initial_docs:
                if doc['id'] in all_gold_ids:
                    gold_docs.append(doc)
                else:
                    non_gold_docs.append(doc)
        
        else:
            # Use answer matching to identify gold documents
            answer_list_string = 'answer_list' if 'answer_list' in gold_inst else 'answers'
            
            for doc in initial_docs:
                contains_answer = False
                
                for answer in gold_inst[answer_list_string]:
                    if answer_list_string == 'answer_list':
                        if has_answer(answer['aliases'], doc['text'], tok):
                            contains_answer = True
                            break
                    else:
                        if has_answer([answer], doc['text'], tok):
                            contains_answer = True
                            break
                
                if contains_answer:
                    gold_docs.append(doc)
                else:
                    non_gold_docs.append(doc)
        
        # Sort gold docs by their original scores (highest first)
        gold_docs.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        # Sort non-gold docs by scores (highest first) 
        non_gold_docs.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        # Combine: gold docs first, then fill with highest scoring non-gold docs
        final_docs = gold_docs.copy()
        
        # Add non-gold docs until we reach final_topk
        remaining_slots = final_topk - len(gold_docs)
        if remaining_slots > 0:
            final_docs.extend(non_gold_docs[:remaining_slots])
        else:
            # If we have more gold docs than final_topk, just take the top final_topk gold docs
            final_docs = final_docs[:final_topk]
        
        # Update scores to reflect new ranking (optional - preserves original scores)
        # If you want new ranking scores: 
        # for i, doc in enumerate(final_docs):
        #     doc['score'] = str(final_topk - i)
        
        reranked_entry['ctxs'] = final_docs
        reranked_documents.append(reranked_entry)
        
        print(f"Question {idx}: {len(gold_docs)} gold docs, {len(final_docs)} total docs")
    
    # Save reranked documents
    write_jsonl(reranked_documents, output_path)
    
    total_gold = sum(len([doc for doc in entry['ctxs'] if is_gold_doc(entry, doc, dataset[i], has_gold_id, tok)]) 
                    for i, entry in enumerate(reranked_documents))
    
    print(f"Reranked {len(reranked_documents)} questions")
    print(f"Total gold documents in final ranking: {total_gold}")
    print(f"Results saved to: {output_path}")

def is_gold_doc(entry, doc, gold_inst, has_gold_id, tokenizer):
    """Helper function to check if a document is gold (for verification)"""
    if has_gold_id:
        all_gold_ids = set()
        if 'positive_ctxs' in gold_inst:
            gold_clusters = gold_inst['positive_ctxs']
        elif 'ground_truths' in gold_inst:
            gold_clusters = gold_inst['ground_truths']
        else:
            return False
        
        for cluster in gold_clusters:
            for gold_doc in cluster:
                all_gold_ids.add(gold_doc['id'])
        
        return doc['id'] in all_gold_ids
    else:
        answer_list_string = 'answer_list' if 'answer_list' in gold_inst else 'answers'
        for answer in gold_inst[answer_list_string]:
            if answer_list_string == 'answer_list':
                if has_answer(answer['aliases'], doc['text'], tokenizer):
                    return True
            else:
                if has_answer([answer], doc['text'], tokenizer):
                    return True
        return False

def rerank_with_gold_priority_0123(retrieved_docs_path, ground_truth_path, output_path, has_gold_id=True, max_gold_docs=3, initial_topk=200, selected_indices=None):
    # Load data
    dataset = read_jsonl(ground_truth_path)
    retrieved_docs = read_jsonl(retrieved_docs_path)
    tok = SimpleTokenizer()
    
    # Validate data lengths
    assert len(dataset) == len(retrieved_docs), f'Length mismatch: {len(dataset)} vs {len(retrieved_docs)}'
    
    # Select subset if specified
    if selected_indices is not None:
        dataset = [dataset[i] for i in selected_indices]
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    reranked_documents = []
    
    for idx, (gold_inst, docs) in enumerate(tqdm(zip(dataset, retrieved_docs), desc="Reranking documents")):
        # Validate questions match
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst['org_q']
        docs_question = docs['question_text'] if 'question_text' in docs else docs['question']
        
        assert gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
        
        # Create new document entry with same structure
        reranked_entry = {
            'question': docs.get('question', gold_question),
            'input': docs.get('input', gold_question), 
            'id': docs.get('id', idx),
            'answers': docs.get('answers', []),
            'ctxs': []
        }
        
        # Copy other fields if they exist
        if 'input_negative_ctxs' in docs:
            reranked_entry['input_negative_ctxs'] = docs['input_negative_ctxs']
        
        # Get initial retrieved documents (top 200)
        initial_docs = docs['ctxs'][:initial_topk]
        
        gold_docs = []
        
        if has_gold_id:
            # Collect all gold IDs
            all_gold_ids = set()
            if 'positive_ctxs' in gold_inst:
                gold_clusters = gold_inst['positive_ctxs']
            elif 'ground_truths' in gold_inst:
                gold_clusters = gold_inst['ground_truths']
            else:
                raise ValueError('No positive_ctxs or ground_truths')
            
            for cluster in gold_clusters:
                for doc in cluster:
                    all_gold_ids.add(doc['id'])
            
            # Find gold documents in retrieved results
            for doc in initial_docs:
                if doc['id'] in all_gold_ids:
                    gold_docs.append(doc)
        
        else:
            # Use answer matching to identify gold documents
            answer_list_string = 'answer_list' if 'answer_list' in gold_inst else 'answers'
            
            for doc in initial_docs:
                contains_answer = False
                
                for answer in gold_inst[answer_list_string]:
                    if answer_list_string == 'answer_list':
                        if has_answer(answer['aliases'], doc['text'], tok):
                            contains_answer = True
                            break
                    else:
                        if has_answer([answer], doc['text'], tok):
                            contains_answer = True
                            break
                
                if contains_answer:
                    gold_docs.append(doc)
        
        # Sort gold docs by their original scores (highest first)
        gold_docs.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        # Only use found gold documents (up to max_gold_docs)
        final_docs = gold_docs[:max_gold_docs]
        
        reranked_entry['ctxs'] = final_docs
        reranked_documents.append(reranked_entry)
        
        print(f"Question {idx}: {len(final_docs)} gold docs used")
    # Save reranked documents
    write_jsonl(reranked_documents, output_path)
    
    total_gold = sum(len(entry['ctxs']) for entry in reranked_documents)
    doc_count_distribution = {}
    for entry in reranked_documents:
        count = len(entry['ctxs'])
        doc_count_distribution[count] = doc_count_distribution.get(count, 0) + 1
    
    print(f"Reranked {len(reranked_documents)} questions")
    print(f"Total gold documents used: {total_gold}")
    print(f"Document count distribution: {doc_count_distribution}")
    print(f"Results saved to: {output_path}") 

def extract_topk_documents(retrieved_docs_path, output_path, topk=3, selected_indices=None):
    """
    Extract just the top-k documents from retrieved documents without any gold document logic.
    
    Args:
        retrieved_docs_path: Path to JSONL file with retrieved documents
        output_path: Path to save the top-k documents
        topk: Number of top documents to keep
        selected_indices: Optional subset of indices to process
    """
    # Load data
    retrieved_docs = read_jsonl(retrieved_docs_path)
    
    # Select subset if specified
    if selected_indices is not None:
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    topk_documents = []
    
    for idx, docs in enumerate(tqdm(retrieved_docs, desc=f"Extracting top-{topk} documents")):
        # Create new document entry with same structure
        topk_entry = {
            'question': docs.get('question', ''),
            'input': docs.get('input', ''), 
            'id': docs.get('id', idx),
            'answers': docs.get('answers', []),
            'ctxs': docs['ctxs'][:topk]  # Just take the top-k documents
        }
        
        # Copy other fields if they exist
        if 'input_negative_ctxs' in docs:
            topk_entry['input_negative_ctxs'] = docs['input_negative_ctxs']
        
        topk_documents.append(topk_entry)
        
        if idx < 5:  # Print first few for debugging
            print(f"Question {idx}: Kept top {len(topk_entry['ctxs'])} documents")
    
    # Save top-k documents
    write_jsonl(topk_documents, output_path)
    
    print(f"Extracted top-{topk} documents for {len(topk_documents)} questions")
    print(f"Results saved to: {output_path}")


def extract_topk_documents_varied(retrieved_docs_path, output_path, ratio_0=25, ratio_1=25, ratio_2=25, ratio_3=25, selected_indices=None):
    """
    Extract variable number of documents from retrieved documents based on specified ratios.
    
    Args:
        retrieved_docs_path: Path to JSONL file with retrieved documents
        output_path: Path to save the documents
        ratio_0: Percentage of queries to use 0 documents (default: 25%)
        ratio_1: Percentage of queries to use 1 document (default: 25%)
        ratio_2: Percentage of queries to use 2 documents (default: 25%)
        ratio_3: Percentage of queries to use 3 documents (default: 25%)
        selected_indices: Optional subset of indices to process
    """
    import random
    
    # Validate ratios sum to 100
    total_ratio = ratio_0 + ratio_1 + ratio_2 + ratio_3
    if total_ratio != 100:
        raise ValueError(f"Ratios must sum to 100, got {total_ratio}")
    
    # Load data
    retrieved_docs = read_jsonl(retrieved_docs_path)
    
    # Select subset if specified
    if selected_indices is not None:
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    # Create list of document counts based on ratios
    total_queries = len(retrieved_docs)
    doc_counts = []
    
    # Calculate number of queries for each document count
    num_0_docs = int(total_queries * ratio_0 / 100)
    num_1_docs = int(total_queries * ratio_1 / 100)
    num_2_docs = int(total_queries * ratio_2 / 100)
    num_3_docs = total_queries - num_0_docs - num_1_docs - num_2_docs  # Remainder goes to 3 docs
    
    # Create list with document counts for each query
    doc_counts.extend([0] * num_0_docs)
    doc_counts.extend([1] * num_1_docs)
    doc_counts.extend([2] * num_2_docs)
    doc_counts.extend([3] * num_3_docs)
    
    # Shuffle to randomize assignment
    random.shuffle(doc_counts)
    
    # Ensure we have exactly the right number of assignments
    assert len(doc_counts) == total_queries, f"Mismatch: {len(doc_counts)} assignments for {total_queries} queries"
    
    variable_documents = []
    count_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for idx, (docs, num_docs) in enumerate(tqdm(zip(retrieved_docs, doc_counts), desc="Extracting variable documents")):
        # Create new document entry with same structure
        variable_entry = {
            'question': docs.get('question', ''),
            'input': docs.get('input', ''), 
            'id': docs.get('id', idx),
            'answers': docs.get('answers', []),
            'ctxs': docs['ctxs'][:num_docs]  # Take the specified number of documents
        }
        
        # Copy other fields if they exist
        if 'input_negative_ctxs' in docs:
            variable_entry['input_negative_ctxs'] = docs['input_negative_ctxs']
        
        variable_documents.append(variable_entry)
        count_distribution[num_docs] += 1
        
        if idx < 5:  # Print first few for debugging
            print(f"Question {idx}: Using {num_docs} documents")
    
    # Save variable documents
    write_jsonl(variable_documents, output_path)
    
    print(f"Extracted variable documents for {len(variable_documents)} questions")
    print(f"Distribution: 0 docs: {count_distribution[0]}, 1 doc: {count_distribution[1]}, 2 docs: {count_distribution[2]}, 3 docs: {count_distribution[3]}")
    print(f"Percentages: 0: {count_distribution[0]/total_queries*100:.1f}%, 1: {count_distribution[1]/total_queries*100:.1f}%, 2: {count_distribution[2]/total_queries*100:.1f}%, 3: {count_distribution[3]/total_queries*100:.1f}%")
    print(f"Results saved to: {output_path}")



def main():

    has_gold_id = False        # whether the gold ids are provided, set False for Answer Recall (AR)
    #input_docs = 0
    file_list = ['/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/qampari_finetuned_base_t200_promote-qwen3.json']
#    gt_data_path = f'../../../../data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus_inp{input_docs}.jsonl'  # ground truth data

    gt_data_path = f'/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'

    use_gold_priority = True  # Set to False for vanilla top-k extraction
    rerank_0123 = False

    for _file in file_list:
        if use_gold_priority and not rerank_0123:
            output_file = f'/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/input_files/qampari_finetuned_base_t200_promote-qwen3_3_unformat.json'
            rerank_with_gold_priority(
                _file,
                gt_data_path,
                output_file,
                has_gold_id=has_gold_id,
                final_topk=3,
                initial_topk=200,
                selected_indices=None
            )
        elif rerank_0123 and use_gold_priority:
            if not has_gold_id:
                 output_file = f'/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/input_files/qampari_finetuned_base_t200_promote-qwen3_0-3_unformat.json'
            else:
                 output_file = f'two_stage/gold_top0123/qampari_standard_only_rand_gold_ER_unformat_0123.json'
            rerank_with_gold_priority_0123(
                 _file,
                 gt_data_path,
                 output_file,
                 has_gold_id=has_gold_id,
                 max_gold_docs=3,  # Instead of final_topk
                 initial_topk=200,
                 selected_indices=None
            )
        else:
            output_file = f'two_stage/vanilla_top0123/qampari_standard_only_rand_vanilla_top0123_unformat_0123.json'
            extract_topk_documents_varied(
                _file,
                output_file,
                ratio_0=25,
                ratio_1=25,
                ratio_2=25,
                ratio_3=25
            )

main()
