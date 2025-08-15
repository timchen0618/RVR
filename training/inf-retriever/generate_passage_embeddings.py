#!/usr/bin/env python3
# Generate passage embeddings using INF-Retriever
# Based on the Contriever passage embedding pipeline

import os
import argparse
import csv
import logging
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import src.slurm
import src.inf_retriever
import src.utils
import src.data
import src.normalize_text


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    
    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p["id"])
            
            # Prepare passage text
            if args.no_title or "title" not in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            
            batch_text.append(text)

            if k % 50 == 0:
                print(f"Embedding passage {k}", flush=True)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                # Tokenize batch
                encoded_batch = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                
                # Get embeddings based on model type
                if hasattr(model, 'base_model'):
                    # INFRetriever wrapper
                    embeddings = model(**encoded_batch)
                elif hasattr(model, 'encoder'):
                    # Wrapped model with encoder attribute
                    embeddings = model.encoder(**encoded_batch)
                else:
                    # Raw transformer model - need to apply pooling
                    outputs = model(**encoded_batch)
                    embeddings = last_token_pool(outputs.last_hidden_state, encoded_batch['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}", flush=True)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    # Load INF-Retriever model
    model, tokenizer, _ = src.inf_retriever.load_retriever(args.model_name_or_path)
    
    # Extract encoder if present (for training mode compatibility)
    if hasattr(model, 'encoder') and not isinstance(model, src.inf_retriever.INFRetriever):
        model = model.encoder

    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    model.eval()
    model = model.cuda()
    print("Model in eval mode")
    
    if not args.no_fp16:
        model = model.half()

    print("Loading passages...", flush=True)
    passages = src.data.load_passages(args.passages)
    print(f"Loaded {len(passages)} passages", flush=True)

    # Handle sharding for distributed processing
    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.", flush=True)

    # Generate embeddings
    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    # Save embeddings
    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.", flush=True)
    
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate passage embeddings using INF-Retriever")

    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="inf_embeddings", help="Directory to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="Prefix for saved embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="ID of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="infly/inf-retriever-v1-1.5b",
        help="Path to INF-Retriever model or HuggingFace model ID"
    )
    parser.add_argument("--no_fp16", action="store_true", help="Inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="Title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="Normalize text before encoding")

    args = parser.parse_args()

    src.slurm.init_distributed_mode(args)

    main(args)