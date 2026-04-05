#!/usr/bin/env python3
"""
LLM-based Reranking Pipeline

Takes a file with queries and ~200 retrieved documents each,
uses an LLM to score relevance, and keeps the top-K (default 100).

Input format: JSONL where each line is:
{
    "question": "...",
    "input": "...",
    "id": 0,
    "answers": [""],
    "ctxs": [
        {"id": "...", "title": "...", "text": "...", "score": "0.77"},
        ...  // ~200 documents
    ]
}

Output format: Same structure but ctxs trimmed to top-K reranked documents,
with an added "rerank_score" field on each doc.
"""

import json
import os
import sys
import argparse
import torch
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
import time
import re

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


class LLMReranker:
    """Rerank documents by relevance using vLLM and a Qwen model."""

    def get_gpu_memory_mb(self):
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except Exception as e:
            print(f"Warning: Could not get GPU memory: {e}", flush=True)
        return 0.0

    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        print(f"Loading vLLM model: {model_name}", flush=True)
        self.total_docs_processed = 0
        self.total_processing_time = 0.0

        memory_before_mb = self.get_gpu_memory_mb()
        print(f"GPU memory before loading: {memory_before_mb:.0f} MB", flush=True)

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="auto",
        )
        time.sleep(2)

        memory_after_mb = self.get_gpu_memory_mb()
        self.llm_memory_gb = (memory_after_mb - memory_before_mb) / 1000
        print(f"GPU memory after loading: {memory_after_mb:.0f} MB "
              f"(model delta: {self.llm_memory_gb:.2f} GB)", flush=True)

        # We ask the LLM to output a single relevance score 0-10.
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            skip_special_tokens=True,
            stop=["\n", "\n\n"],
        )
        print("Model loaded successfully!\n", flush=True)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def create_rerank_prompt(self, query: str, document: Dict[str, Any]) -> str:
        title = document.get('title', 'No title')
        text = document.get('text', '')
        # Truncate very long texts to stay within context window
        # max_text_chars = 2000
        # if len(text) > max_text_chars:
        #     text = text[:max_text_chars] + "..."

        return (
            "You are a relevance assessment assistant. "
            "Determine whether the following document is relevant to the question.\n\n"
            "Respond with ONLY 'Yes' or 'No'. No explanation.\n\n"
            f"Question: {query}\n\n"
            f"Document Title: {title}\n"
            f"Document Text: {text}\n\n"
            "Relevant:"
        )

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------
    @staticmethod
    def parse_score(raw: str) -> float:
        """Extract a binary score from the LLM output (Yes=1, No=0)."""
        raw = raw.strip().lower()
        if raw.startswith('yes'):
            return 1.0
        elif raw.startswith('no'):
            return 0.0
        # Fallback: check for numeric
        m = re.search(r'[0-9]', raw)
        if m:
            return 1.0 if int(m.group()) > 0 else 0.0
        return 0.0  # unparseable → not relevant

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------
    def score_documents(
        self, query: str, documents: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Score all documents for a query. Returns list of (doc, score) pairs."""
        scored: List[Tuple[Dict[str, Any], float]] = []
        start = time.time()

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            print(
                f"  Scoring batch {i // batch_size + 1}: "
                f"docs {i + 1}-{min(i + batch_size, len(documents))}",
                flush=True,
            )

            prompts = [self.create_rerank_prompt(query, doc) for doc in batch]

            try:
                outputs = self.llm.generate(prompts, self.sampling_params)
                for j, output in enumerate(outputs):
                    raw = output.outputs[0].text.strip()
                    score = self.parse_score(raw)
                    scored.append((batch[j], score))
            except Exception as e:
                print(f"  Error in batch {i // batch_size + 1}: {e}", flush=True)
                for doc in batch:
                    scored.append((doc, 0.0))

        elapsed = time.time() - start
        self.total_docs_processed += len(documents)
        self.total_processing_time += elapsed
        throughput = len(documents) / elapsed if elapsed > 0 else 0
        print(f"  Scored {len(documents)} docs in {elapsed:.1f}s "
              f"({throughput:.1f} docs/sec)", flush=True)
        return scored

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def print_stats(self):
        avg = (self.total_docs_processed / self.total_processing_time
               if self.total_processing_time > 0 else 0)
        print("\n" + "=" * 70)
        print("RERANKER EFFICIENCY STATS")
        print("=" * 70)
        print(f"Total documents scored : {self.total_docs_processed}")
        print(f"Total time             : {self.total_processing_time:.2f}s")
        print(f"Average throughput     : {avg:.2f} docs/sec")
        print(f"LLM memory (delta)     : {self.llm_memory_gb:.2f} GB")
        print("=" * 70 + "\n", flush=True)


# ======================================================================
# Processing
# ======================================================================

def process_file(
    reranker: LLMReranker,
    input_file: str,
    output_file: str,
    top_k: int = 100,
    batch_size: int = 32,
):
    print("\n" + "=" * 70)
    print("LLM RERANKING PIPELINE")
    print("=" * 70)
    print(f"Input  : {input_file}")
    print(f"Output : {output_file}")
    print(f"Top-K  : {top_k}")
    print("=" * 70 + "\n")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Load — support both JSON array and JSONL
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    if raw.startswith('['):
        entries = json.loads(raw)
    else:
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]

    print(f"Loaded {len(entries)} queries\n")

    processed = []
    stats = {
        'total_queries': len(entries),
        'total_input_docs': 0,
        'total_output_docs': 0,
    }

    for idx, entry in enumerate(entries, 1):
        query = entry.get('question', entry.get('input', ''))
        documents = entry.get('ctxs', [])[:400]

        print(f"[{idx}/{len(entries)}] Query: {query[:100]}...")
        print(f"  Input documents: {len(documents)}")

        stats['total_input_docs'] += len(documents)

        if not documents:
            processed.append(entry)
            continue

        # Score all documents
        scored = reranker.score_documents(query, documents, batch_size)

        # Sort by LLM score descending, break ties by original retrieval score
        scored.sort(
            key=lambda pair: (pair[1], float(pair[0].get('score', 0))),
            reverse=True,
        )

        # Keep top-K and attach rerank_score
        top_docs = []
        for doc, llm_score in scored[:top_k]:
            doc_copy = dict(doc)
            doc_copy['rerank_score'] = llm_score
            top_docs.append(doc_copy)

        stats['total_output_docs'] += len(top_docs)
        print(f"  Kept top {len(top_docs)} documents "
              f"(score range: {scored[0][1]:.0f} – "
              f"{scored[min(top_k, len(scored)) - 1][1]:.0f})\n")

        new_entry = {
            "question": entry.get('question', ''),
            "input": entry.get('input', ''),
            "id": entry.get('id', idx - 1),
            "answers": entry.get('answers', []),
            "ctxs": top_docs,
        }
        processed.append(new_entry)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Queries          : {stats['total_queries']}")
    print(f"Total input docs : {stats['total_input_docs']}")
    print(f"Total output docs: {stats['total_output_docs']}")
    print(f"Saved to         : {output_file}")
    print("=" * 70 + "\n")

    reranker.print_stats()


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rerank retrieved documents with an LLM and keep the top-K"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input file (JSON array or JSONL)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSON file")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of documents to keep after reranking (default: 100)")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="vLLM model for scoring")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for LLM inference (default: 32)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", flush=True)
        sys.exit(1)

    print("\nInitializing reranker...")
    reranker = LLMReranker(model_name=args.model_name)

    process_file(
        reranker=reranker,
        input_file=args.input_file,
        output_file=args.output_file,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()