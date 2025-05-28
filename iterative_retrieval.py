import json
import argparse
from src.data import load_data, prepare_retrieval_prompts, get_instruction
from src.utils.file_utils import write_jsonl
from src.retrieval.bm25 import BM25Retriever, BM25LuceneSearcher
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from inferece import run_inference, load_lm
from tqdm import tqdm
from src.utils.logger import Logger
from src.utils.schema import RetrievalSchema
from pathlib import Path

logger = Logger('IterativeRetrieval', 'iterative_retrieval.log').get_logger()


def add_question(example):
    question_str = 'question'
    if 'question_text' in example:
        question_str = 'question_text'
    elif 'query' in example:
        question_str = 'query'
    example['question'] = example[question_str]
    return example

def filter_docs(retrieved_doc_step, docs):
    filtered_docs = []
    for doc in retrieved_doc_step:
        if doc['text'] not in docs:
            filtered_docs.append(doc)
    assert len(filtered_docs) > 0
    return filtered_docs

def process_query(query):
    try:
        q = json.loads(query)
    except:
        print('error')
        return query
    return q['generated_query']
    

def main(args):
    """
        Given a query, first retrieve one document. 
        Then, generate a new query based on the retrieved document.
        Repeat the process for k times.
    """
    # load LM
    logger.info(f'Loading model from {args.model_path}')
    model, tokenizer = load_lm(args)
    
    # load retriever
    # retriever = load_retriever()
    logger.info('Loading BM25 retriever')
    retriever = BM25LuceneSearcher(corpus_path='')

    # load data
    logger.info(f'Loading {args.dataset} dataset')
    dataset = load_data(dataset_name=args.dataset, retrieved_doc_type='none', split=args.split)
    instruction = get_instruction(args.instruction_file)
    if Path(args.instruction_file).stem == 'instruction_retrieval_refined':
        schema = RetrievalSchema
    else:
        schema = None
    
    logger.info('Start iterative retrieval')
    results = []
    # inference loop
    for example in tqdm(dataset):
        doc_texts = []
        docs = []
        
        # preprocessing 
        example = add_question(example)
        query = example['question']
        org_question = example['question']

        # inference for each example
        for step in range(args.top_k):
            # retrieve
            retrieved_doc_step = retriever.retrieve(query, n=args.top_k*2)
            filtered_docs = filter_docs(retrieved_doc_step, doc_texts)
            doc_texts += [doc['text'] for doc in filtered_docs[:1]]
            docs += filtered_docs[:1]

            # regenerate query based on already retrieved documents
            _input = prepare_retrieval_prompts(org_question, doc_texts, str(step+1), tokenizer, instruction, support_system_prompt=args.support_system_prompt)
            # print('input', _input)
            outputs = run_inference(model, [_input], args, schema=schema, use_tqdm=False)

            # print(outputs[0])
            # update query
            if isinstance(schema, RetrievalSchema):
                query = process_query(outputs[0])
            else:
                query = outputs[0]

            # query = model.generate_query(_input)

        # append result to outputs
        results.append({'question': example['question'], 'ctxs': docs, 'answers': [""], 'id': example['qid']})

    # save results   
    write_jsonl(results, f'retrieval_outputs/{args.dataset}/iterative_retrieval_refined.jsonl')
            

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iterative Retrieval Script')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='qampari')
    parser.add_argument('--split', type=str, help='Dataset split', default='dev')
    parser.add_argument('--top_k', type=int, help='Number of iterations', default=5)
    parser.add_argument('--model_path', type=str, help='Path to the model', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--support_system_prompt', action='store_true', help='Support system prompt')
    parser.add_argument('--instruction_file', type=str, help='Instruction file', default='instructions/instruction_retrieval_refined.txt')

    # generation parameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    parser.add_argument("--use_outline", action='store_true', help='Use outline')
    
    args = parser.parse_args()
    main(args)