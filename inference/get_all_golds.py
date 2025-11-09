import json
from sklearn.metrics import precision_score, recall_score, f1_score
import regex
import string
from tqdm import tqdm
import unicodedata
import pytrec_eval
from nltk import word_tokenize

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
    """Read a JSONL file, skipping malformed lines."""
    data = []
    error_count = 0
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                example = json.loads(line)
                data.append(example)
            except json.JSONDecodeError:
                error_count += 1
                print(f"Warning: Skipping malformed JSON at line {line_num} in {file_path}")
                continue
    if error_count > 0:
        print(f"Total malformed lines skipped: {error_count}")
    return data

def write_jsonl(data, file_path):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def extract_all_gold_documents(retrieved_docs_path, ground_truth_path, output_path, 
                              has_gold_id=False, initial_topk=100, selected_indices=None):
    """
    Extract ALL gold documents from retrieved documents, maintaining exact same JSON structure.
    
    Args:
        retrieved_docs_path: Path to JSONL file with retrieved documents
        ground_truth_path: Path to JSONL file with ground truth data
        output_path: Path to save ALL gold documents (same format as top-k)
        has_gold_id: Whether gold document IDs are provided
        initial_topk: Number of documents to consider from retrieved (100)
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
    
    all_gold_documents = []
    
    # Statistics tracking
    stats = {
        'total_questions': 0,
        'questions_with_gold': 0,
        'total_gold_found': 0,
        'distribution': {}
    }
    
    for idx, (gold_inst, docs) in enumerate(tqdm(zip(dataset, retrieved_docs), desc="Extracting ALL gold documents")):
        # Validate questions match
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst['org_q']
        docs_question = docs['question_text'] if 'question_text' in docs else docs['question']
        
        # gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
        
        # Create new document entry with EXACT same structure as original
        all_gold_entry = {
            'question': docs.get('question', gold_question),
            'input': docs.get('input', gold_question), 
            'id': docs.get('id', idx),
            'answers': docs.get('answers', []),
            'ctxs': []  # Will contain ALL gold documents
        }
        
        # Copy ALL other fields if they exist (maintain exact structure)
        for key, value in docs.items():
            if key not in ['question', 'input', 'id', 'answers', 'ctxs']:
                all_gold_entry[key] = value
        
        # Get initial retrieved documents
        initial_docs = docs['ctxs'][:initial_topk]
        
        all_gold_docs = []
        
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
            
            # Find ALL gold documents in retrieved results
            for doc in initial_docs:
                if doc['id'] in all_gold_ids:
                    all_gold_docs.append(doc)
        
        else:
            # Use answer matching to identify ALL gold documents
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
                    all_gold_docs.append(doc)
        
        # Sort gold docs by their original scores (highest first)
        all_gold_docs.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        # Store ALL gold documents (not just top 3)
        all_gold_entry['ctxs'] = all_gold_docs
        
        all_gold_documents.append(all_gold_entry)
        
        # Update statistics
        stats['total_questions'] += 1
        num_gold = len(all_gold_docs)
        if num_gold > 0:
            stats['questions_with_gold'] += 1
        stats['total_gold_found'] += num_gold
        
        if num_gold not in stats['distribution']:
            stats['distribution'][num_gold] = 0
        stats['distribution'][num_gold] += 1
        
        if idx < 5:  # Print first few for debugging
            print(f"Question {idx}: Found {len(all_gold_docs)} gold documents")
    
    # Save all gold documents (same format as original)
    write_jsonl(all_gold_documents, output_path)
    
    # Print statistics
    print(f"\n=== ALL GOLD DOCUMENTS EXTRACTION STATISTICS ===")
    print(f"Total questions processed: {stats['total_questions']}")
    print(f"Questions with at least 1 gold doc: {stats['questions_with_gold']}")
    print(f"Percentage with gold docs: {stats['questions_with_gold']/stats['total_questions']*100:.1f}%")
    print(f"Total gold documents found: {stats['total_gold_found']}")
    print(f"Average gold docs per question: {stats['total_gold_found']/stats['total_questions']:.2f}")
    
    print(f"\nDistribution of gold docs found per question:")
    for count in sorted(stats['distribution'].keys()):
        num_questions = stats['distribution'][count]
        percentage = num_questions / stats['total_questions'] * 100
        print(f"  {count} gold docs: {num_questions} questions ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Format: Exact same structure as input, but ctxs contains ALL gold documents")
    
    return stats


def main():

    
    has_gold_id = False  # False = AR
    
    retrieved_docs_file = '/scratch/dq2024/diverse_retriever/data/contriever_training_file_retrieved.json'
   # gt_data_path = '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    gt_data_path = '/scratch/dq2024/diverse_retriever/train_data.jsonl'
    
    output_file = '/scratch/dq2024/diverse_retriever/data/contriever_training_file_retrieved_all_golds.json'
    
    stats = extract_all_gold_documents(
        retrieved_docs_path=retrieved_docs_file,
        ground_truth_path=gt_data_path,
        output_path=output_file,
        has_gold_id=has_gold_id,
        initial_topk=100,
        selected_indices=None
    )
    
    print(f"File saved: {output_file}")
    print(f"Total gold documents: {stats['total_gold_found']}")
    print(f"Questions with gold: {stats['questions_with_gold']}/{stats['total_questions']}")
    


if __name__ == "__main__":
    main()