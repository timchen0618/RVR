
# /scratch/dq2024/diverse_retriever/inference/contriever/base/qampari_corpus_finetuned_base_sample_0_t200.json
# /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t200.json
# /scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_corpus_finetuned_base_t200.json

'''

    --input  /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t200.json \
    --output /scratch/dq2024/diverse_retriever/eval/infly/base/qampari_corpus_finetuned_base_t100_mmr_from_200.json \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_base_finetuned_t0.05_lr0.00001_ws1000_bs48_acc2_gradchkpt/checkpoint/best_model \
    
--input /scratch/dq2024/diverse_retriever/inference/contriever/base/qampari_corpus_finetuned_base_sample_0_t200.json \
--output /scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100_mmr_from_200.json \
--model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_base_finetuned_only_random_sample_0/checkpoint/best_model \

--input /scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_corpus_finetuned_base_t200.json \
--output /scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_corpus_finetuned_base_t100_mmr_from_200.json \
--model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_base_finetuned_t0.05_lr0.00001_ws1000_bs48_acc2_rightpad_gradchkpt/checkpoint/best_model \

'''

import os
import sys
import json
import argparse
import numpy as np
import torch
import transformers
import sys
import time

def load_model(model_type, model_name_or_path, no_fp16=False):
    if model_type == "contriever":
        sys.path.insert(0, "/scratch/dq2024/diverse_retriever/training/contriever")
        import src.contriever as retriever_module
        pooling = "average"
    elif model_type == "infly":
        sys.path.insert(0, "/scratch/dq2024/diverse_retriever/training/inf_retriever")
        import src.inf_retriever as retriever_module
        pooling = "last_token"
    elif model_type == "qwen":
        sys.path.insert(0, "/scratch/dq2024/diverse_retriever/training/qwen2")
        import src.qwen_retriever as retriever_module
        pooling = "last_token"
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from contriever, infly, qwen.")

    print(f"Loading {model_type} model from: {model_name_or_path}", flush=True)
    model, tokenizer, retriever_model_id = retriever_module.load_retriever(
        model_name_or_path, pooling=pooling
    )
    if args.training_mode == "base" and hasattr(model, "encoder"):
        model = model.encoder
    model.eval()
    model = model.cuda()
    if not no_fp16:
        model = model.half()
    print(f"Model loaded ({retriever_model_id})", flush=True)
    return model, tokenizer


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


@torch.no_grad()
def embed_texts(texts, model, tokenizer, model_type, batch_size=64, max_length=512, normalize=None):
    """
    Encode a list of texts using the retriever model.
    Mirrors embed_batch_texts from passage_retrieval.py.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        encoded = {k: v.cuda() for k, v in encoded.items()}

        if model_type == "contriever":
            # Contriever: normalize=False by default, we normalize manually after
            output = model(**encoded)
        else:
            # Infly / Qwen: normalize=True is the default
            if normalize is not None and 'normalize' in model.forward.__code__.co_varnames:
                output = model(**encoded, normalize=normalize)
            else:
                output = model(**encoded)

        all_embeddings.append(output.float())  # cast from fp16/bf16 to fp32

    return torch.cat(all_embeddings, dim=0)  # (N, D)


def prepare_query_texts(queries, model_type):
    """
    Apply the correct query formatting for each model type.
    - contriever / standard: plain query
    - infly / qwen / base:   wrap with get_detailed_instruct
    """
    if model_type == "contriever":
        return queries
    else:
        task = 'Given a query, retrieve relevant passages that answer the query'
        return [get_detailed_instruct(task, q) for q in queries]


def mmr(
    query_emb: np.ndarray,       # (D,)
    doc_embs: np.ndarray,        # (N, D)  — already L2-normalized
    docs: list,
    top_k: int = 100,
    lambda_param: float = 0.5,
) -> list:
    """
    Greedy MMR selection.

    MMR = argmax_{d_i in C\S} [
        lambda * sim(d_i, q) - (1 - lambda) * max_{d_j in S} sim(d_i, d_j)
    ]

    Both similarities are cosine (dot product on L2-normalized vectors).
    """
    top_k = min(top_k, len(docs))
    n = len(docs)

    # L2-normalize so dot product = cosine similarity
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    d_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)

    sim_to_query = d_norms @ q_norm          # (N,)
    sim_doc_doc  = d_norms @ d_norms.T       # (N, N)

    selected  = []
    remaining = list(range(n))

    for _ in range(top_k):
        best_idx   = None
        best_score = float("-inf")

        for i in remaining:
            relevance = float(sim_to_query[i])

            if not selected:
                redundancy = 0.0
            else:
                redundancy = float(np.max(sim_doc_doc[i, selected]))

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx   = i

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [dict(docs[i], mmr_score=round(float(sim_to_query[i]), 6)) for i in selected]


def apply_mmr_to_file(args):
    # Load model
    model, tokenizer = load_model(args.model_type, args.model_name_or_path, args.no_fp16)

    # Load entries
    all_entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))
    print(f"Loaded {len(all_entries)} entries", flush=True)

    total_before = 0
    total_after  = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(all_entries):
            query = entry.get("question", entry.get("input", ""))
            ctxs  = entry.get("ctxs", [])
            total_before += len(ctxs)

            if not ctxs:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            # ── Encode query ──────────────────────────────────────────────
            query_text  = prepare_query_texts([query], args.model_type)
            query_emb   = embed_texts(
                query_text, model, tokenizer, args.model_type,
                batch_size=1, max_length=args.question_maxlength
            ).cpu().numpy()[0]   # (D,)

            # ── Encode documents ──────────────────────────────────────────
            doc_texts = [
                (d.get("title", "") + " " + d.get("text", "")).strip()
                for d in ctxs
            ]
            doc_embs = embed_texts(
                doc_texts, model, tokenizer, args.model_type,
                batch_size=args.per_gpu_batch_size,
                max_length=args.question_maxlength
            ).cpu().numpy()      # (N, D)

            # ── MMR selection ─────────────────────────────────────────────
            total_mmr_time = 0.0
            start = time.time()
            selected = mmr(
                query_emb=query_emb,
                doc_embs=doc_embs,
                docs=ctxs,
                top_k=args.top_k,
                lambda_param=args.lambda_param,
            )
            total_mmr_time += time.time() - start
            avg_mmr_time = total_mmr_time / len(all_entries)
            print(f"Avg MMR time per query : {avg_mmr_time:.4f} s/q")
            total_after += len(selected)


            new_entry = dict(entry)
            new_entry["ctxs"] = selected
            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(all_entries)} entries", flush=True)

    avg_before = total_before / len(all_entries) if all_entries else 0
    avg_after  = total_after  / len(all_entries) if all_entries else 0

    print(f"\n=== MMR Complete ===")
    print(f"Model type             : {args.model_type}")
    print(f"Model path             : {args.model_name_or_path}")
    print(f"Total entries          : {len(all_entries)}")
    print(f"Avg ctxs before MMR    : {avg_before:.1f}")
    print(f"Avg ctxs after MMR     : {avg_after:.1f}")
    print(f"Lambda                 : {args.lambda_param}")
    print(f"Top-k selected         : {args.top_k}")
    print(f"Output written to      : {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",               required=True,  type=str)
    parser.add_argument("--output",              required=True,  type=str)
    parser.add_argument("--model_name_or_path",  required=True,  type=str)
    parser.add_argument("--model_type",          required=True,  type=str,
                        choices=["contriever", "infly", "qwen"],
                        help="Which retriever architecture to use")
    parser.add_argument("--top_k",              default=100,    type=int)
    parser.add_argument("--lambda_param",        default=0.5,    type=float)
    parser.add_argument("--per_gpu_batch_size",  default=64,     type=int)
    parser.add_argument("--question_maxlength",  default=512,    type=int)
    parser.add_argument("--no_fp16",             action="store_true")
    parser.add_argument("--training_mode", default="base", type=str,
                    choices=["base", "standard"])
    args = parser.parse_args()

    apply_mmr_to_file(args)