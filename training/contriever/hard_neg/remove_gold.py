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
    import json
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def write_jsonl(data, file_path):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def remove_all_gold_documents(retrieved_docs_path, ground_truth_path, output_path, selected_indices=None):
    """
    Remove all gold documents from retrieved results using both Evidence Recall and Answer Recall criteria.
    
    Args:
        retrieved_docs_path: Path to JSONL file with retrieved documents
        ground_truth_path: Path to JSONL file with ground truth data
        output_path: Path to save the cleaned documents (with gold docs removed)
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
    
    cleaned_documents = []
    removal_stats = {
        'total_questions': 0,
        'evidence_recall_removed': 0,
        'answer_recall_removed': 0,
        'total_docs_before': 0,
        'total_docs_after': 0
    }
    
    for idx, (gold_inst, docs) in enumerate(tqdm(zip(dataset, retrieved_docs), desc="Removing gold documents")):
        # Validate questions match
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst.get('org_q', '')
        docs_question = docs['question_text'] if 'question_text' in docs else docs.get('question', '')
        
        if gold_question and docs_question:
            assert gold_question == docs_question, f'Questions do not match: {gold_question} vs {docs_question}'
        
        # Create new document entry with same structure
        cleaned_entry = {
            'question': docs.get('question', gold_question),
            'input': docs.get('input', gold_question), 
            'id': docs.get('id', idx),
            'answers': docs.get('answers', []),
            'ctxs': []
        }
        
        # Copy other fields if they exist
        for field in ['input_negative_ctxs', 'question_text', 'qid']:
            if field in docs:
                cleaned_entry[field] = docs[field]
        
        # Collect all gold document IDs (Evidence Recall)
        gold_ids = set()
        
        # Check positive_ctxs
        if 'positive_ctxs' in gold_inst:
            gold_clusters = gold_inst['positive_ctxs']
            for cluster in gold_clusters:
                for doc in cluster:
                    gold_ids.add(doc['id'])

        # Check ground_truths  
        if 'ground_truths' in gold_inst:
            gold_clusters = gold_inst['ground_truths']
            for cluster in gold_clusters:
                for doc in cluster:
                    gold_ids.add(doc['id'])
        
        # Filter retrieved documents
        filtered_ctxs = []
        evidence_removed = 0
        answer_removed = 0
        
        for doc in docs['ctxs']:
            should_remove = False
            
            # Check Evidence Recall: Skip if document ID matches
            if doc['id'] in gold_ids:
                should_remove = True
                evidence_removed += 1
            
            # Check Answer Recall: Skip if document contains answers
            if 'answer_list' in gold_inst:
                contains_answer = False
                for answer in gold_inst['answer_list']:
                    if has_answer(answer['aliases'], doc['text'], tok):
                        contains_answer = True
                        break
                
                if contains_answer:
                    should_remove = True
                    answer_removed += 1
            
            # Keep document only if it's not gold by either criterion
            if not should_remove:
                filtered_ctxs.append(doc)
        
        cleaned_entry['ctxs'] = filtered_ctxs
        cleaned_documents.append(cleaned_entry)
        
        removal_stats['total_questions'] += 1
        removal_stats['evidence_recall_removed'] += evidence_removed
        removal_stats['answer_recall_removed'] += answer_removed
        removal_stats['total_docs_before'] += len(docs['ctxs'])
        removal_stats['total_docs_after'] += len(filtered_ctxs)
        
        if idx < 5:
            print(f"Question {idx}: Removed {evidence_removed} evidence docs, {answer_removed} answer docs, kept {len(filtered_ctxs)} docs")
    
    write_jsonl(cleaned_documents, output_path)
    
    print(f"\n=== Gold Document Removal Summary ===")
    print(f"Total questions processed: {removal_stats['total_questions']}")
    print(f"Documents removed by Evidence Recall: {removal_stats['evidence_recall_removed']}")
    print(f"Documents removed by Answer Recall: {removal_stats['answer_recall_removed']}")
    print(f"Total documents before: {removal_stats['total_docs_before']}")
    print(f"Total documents after: {removal_stats['total_docs_after']}")
    print(f"Total documents removed: {removal_stats['total_docs_before'] - removal_stats['total_docs_after']}")
    print(f"Removal rate: {((removal_stats['total_docs_before'] - removal_stats['total_docs_after']) / removal_stats['total_docs_before'] * 100):.2f}%")
    print(f"Average docs per question before: {removal_stats['total_docs_before'] / removal_stats['total_questions']:.2f}")
    print(f"Average docs per question after: {removal_stats['total_docs_after'] / removal_stats['total_questions']:.2f}")
    print(f"Results saved to: {output_path}")


def main():
    """
    Main function to remove gold documents from retrieved results.
    """
    
    # Configuration
    input_file = 'qampari_corpus_base_finetuned_hard_neg_train.json'  
    ground_truth_file = 'train_data.jsonl'  
    
    # Output files for different removal strategies
    output_all_removed = 'qampari_corpus_base_finetuned_hard_neg.json'
    
    
    remove_all_gold_documents(
        input_file,
        ground_truth_file,
        output_all_removed,
        selected_indices=None
    )
    
    print("\n=== All removal operations completed! ===")

if __name__ == "__main__":
    main()