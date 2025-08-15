#!/usr/bin/env python3
# Retrieval script for INF-Retriever models
# Based on the Contriever retrieval pipeline

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import src.index
import src.inf_retriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings by taking the last token's representation"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for better retrieval performance"""
    return f'Instruct: {task_description}\nQuery: {query}'


@torch.no_grad()
def embed_batch_texts(args, texts, model, tokenizer):
    embeddings, batch_texts = [], []
    
    for k, _text in enumerate(texts):
        if args.lowercase:
            _text = _text.lower()
        if args.normalize_text:
            _text = src.normalize_text.normalize(_text)
        batch_texts.append(_text)

        if len(batch_texts) == args.per_gpu_batch_size or k == len(texts) - 1:
            encoded_batch = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                max_length=args.question_maxlength,
                padding=True,
                truncation=True,
            )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            output = model(**encoded_batch)
            embeddings.append(output)
            batch_texts = []

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


@torch.no_grad()
def embed_queries(args, queries, input_negatives, model, tokenizer):
    if args.training_mode == "base" or args.training_mode == "standard" or args.training_mode == "standard_concat":
        return embed_queries_standard(args, queries, model, tokenizer)
    elif args.training_mode == "subtraction":
        return embed_queries_subtraction(args, queries, input_negatives, model, tokenizer)
    elif args.training_mode == "gru":
        return embed_queries_gru(args, queries, input_negatives, model, tokenizer)
    elif args.training_mode == "linear_projection":
        return embed_queries_linear_projection(args, queries, input_negatives, model, tokenizer)
    else:
        raise ValueError(f"Invalid training mode: {args.training_mode}")


@torch.no_grad()
def embed_queries_standard(args, queries, model, tokenizer):
    model.eval()
    if args.training_mode == "base":
        if hasattr(model, 'encoder'):
            embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
        else:
            embeddings = embed_batch_texts(args, queries, model, tokenizer)
    else:
        embeddings = embed_batch_texts(args, queries, model, tokenizer)
    
    print(f"Questions embeddings shape: {embeddings.size()}")
    return embeddings.cpu().numpy()


@torch.no_grad()
def embed_queries_subtraction(args, queries, input_negatives, model, tokenizer):
    model.eval()
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]
    
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        if _len > 0:
            embeddings[i] = embeddings[i] - inpn_embeddings[start:end].sum(dim=0)
        start = end
    return embeddings.cpu().numpy()


@torch.no_grad()
def embed_queries_gru(args, queries, input_negatives, model, tokenizer):
    model.eval()
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]
    
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")
    
    gru = model.gru
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        if _len > 0:
            _, hn = gru(inpn_embeddings[start:start+_len], embeddings[i].unsqueeze(0))
            embeddings[i] = hn[-1]
        start = end
    return embeddings.cpu().numpy()


@torch.no_grad()
def embed_queries_linear_projection(args, queries, input_negatives, model, tokenizer):
    model.eval()
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]
    
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")
    
    linear = model.linear
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        if _len > 0:
            doc_emb = inpn_embeddings[start:start+_len].mean(dim=0)
            q_doc_emb = torch.cat([embeddings[i].unsqueeze(0), doc_emb.unsqueeze(0)], dim=1)
            embeddings[i] = linear(q_doc_emb).squeeze(0)
        start = end
    return embeddings.cpu().numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}", flush=True)
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        if 'ctxs' not in d:
            d['ctxs'] = []
        d["ctxs"] += [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        error_count = 0
        with open(data_path, "r") as fin:
            for line_num, line in enumerate(fin, 1):
                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Warning: Skipping malformed JSON at line {line_num} in {data_path}")
                    if error_count <= 5:  
                        print(f"Problematic line preview (first 200 chars): {line[:200]}...")
                    continue
        
        if error_count > 0:
            print(f"Total malformed lines skipped: {error_count}", flush=True)
    return data


def main(args):
    print(f"Loading INF-Retriever model from: {args.model_name_or_path}", flush=True)
    model, tokenizer, _ = src.inf_retriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    # INF-Retriever uses 1536 dimensions by default
    if args.projection_size == 768:  # Default from Contriever
        print("Warning: INF-Retriever uses 1536 dimensions. Updating projection_size.")
        args.projection_size = 1536
    
    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    # Index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}", flush=True)
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.", flush=True)
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # Load passages
    passages = src.data.load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    # Load data
    print(f"Retrieving for file: {args.data}", flush=True)
    data = load_data(args.data)
    output_path = os.path.join(args.output_dir, os.path.basename(args.outfile))

    # Embed the queries
    queries = [ex[args.question_key] for ex in data]
    print(f'Running retrieval with mode: {args.training_mode}', flush=True)
    
    input_negatives = []
    if args.training_mode != "standard" and (not args.autoregressive) and args.training_mode != "base":
        for example in data:
            if 'input_negative_ctxs' in example:
                input_negatives.append([
                    n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] 
                    for n in example['input_negative_ctxs']
                ])
            else:
                input_negatives.append([])
                
    elif args.training_mode == "standard_concat":
        for example in data:
            if 'input_negative_ctxs' in example:
                input_negatives.append([
                    n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] 
                    for n in example['input_negative_ctxs']
                ])
            else:
                input_negatives.append([])
        queries = ['Question: ' + ex[args.question_key] + '\n\n' + 'Documents:' + '\n'.join(inpn) 
                   for ex, inpn in zip(data, input_negatives)]
    
    questions_embedding = embed_queries(args, queries, input_negatives, model, tokenizer)

    if not args.autoregressive:
        # Get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.", flush=True)
        add_passages(data, passage_id_map, top_ids_and_scores)
    else:
        # Autoregressive retrieval
        num_docs = 0
        while True:
            start_time_retrieval = time.time()
            top_ids_and_scores = index.search_knn(questions_embedding, args.step)
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.", flush=True)
            num_docs += args.step
            if num_docs >= args.n_docs:
                break
            
            # Add the passages to ctxs
            add_passages(data, passage_id_map, top_ids_and_scores)
            for inst in data:
                inst['input_negative_ctxs'] = inst['ctxs']
                assert len(inst['input_negative_ctxs']) == num_docs
            
            # Re-embed the query
            print(f'Re-embedding with mode: {args.training_mode}', flush=True)
            input_negatives = []
            for example in data:
                input_negatives.append([
                    n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] 
                    for n in example['input_negative_ctxs']
                ])
            if args.training_mode == "standard":
                assert len(data) == len(input_negatives)
                queries = ['Question: ' + ex[args.question_key] + '\n\n' + 'Documents:' + '\n'.join(inpn) 
                           for ex, inpn in zip(data, input_negatives)]
            else:
                queries = [ex[args.question_key] for ex in data]
                
            questions_embedding = embed_queries(args, queries, input_negatives, model, tokenizer)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval using INF-Retriever")

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing questions and answers",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="infly/inf-retriever-v1-1.5b",
        help="Path to INF-Retriever model or HuggingFace model ID"
    )
    parser.add_argument("--no_fp16", action="store_true", help="Inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=1536, help="Embedding dimension (1536 for INF-Retriever)")
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="Normalize text")
    parser.add_argument("--use_instruction", action="store_true", default=True, help="Use instruction for queries")
    
    parser.add_argument("--outfile", type=str, default="output.jsonl")
    parser.add_argument("--training_mode", type=str, default="standard")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--autoregressive", action="store_true", help="Autoregressive retrieval")
    parser.add_argument("--step", type=int, default=3, help="Number of documents to retrieve per step")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)