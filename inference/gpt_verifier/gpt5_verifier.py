import json
import openai
from typing import List, Dict, Any, Tuple
import time
import argparse
import os
from datetime import datetime
import asyncio
import aiohttp

class DocumentVerifier:
    def __init__(self, api_key: str, model: str = "gpt-5-nano", max_concurrent: int = 50, batch_size: int = 5):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.base_url = "https://api.openai.com/v1/responses"
        
    def create_batch_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create a batch verification prompt"""
        query = str(query).encode('ascii', 'ignore').decode('ascii', 'ignore')
        
        parts = []
        parts.append("You are a document relevance verifier. Your task is to determine if given documents contain direct answers to a specific question.")
        parts.append("")
        parts.append(f"Question: {query}")
        parts.append("")
        parts.append("Documents to evaluate:")
        
        for i, doc in enumerate(documents):
            title = str(doc.get('title', 'No title')).encode('ascii', 'ignore').decode('ascii', 'ignore')
            text = str(doc.get('text', ''))[:1000].encode('ascii', 'ignore').decode('ascii', 'ignore')
            parts.append("")
            parts.append(f"Document {i+1}:")
            parts.append(f"Title: {title}")
            parts.append(f"Text: {text}...")
        
        parts.append("")
        parts.append("Instructions:")
        parts.append("- For each document, answer with ONLY YES if it directly answers the question or NO if it doesn't")
        parts.append("- Provide your answers as a comma-separated list in the exact order of the documents")
        parts.append("- Example format: YES,NO,YES,NO,YES")
        parts.append("- Be strict in your evaluation")
        parts.append("")
        parts.append("Your answers (comma-separated):")
        
        return "\n".join(parts).encode('ascii', 'ignore').decode('ascii', 'ignore')

    async def verify_batch_async(self, session: aiohttp.ClientSession, 
                                  query: str, documents: List[Dict[str, Any]],
                                  batch_idx: int, semaphore: asyncio.Semaphore) -> Tuple[int, List[bool]]:
        """Verify a batch of documents asynchronously"""
        async with semaphore:
            prompt = self.create_batch_prompt(query, documents)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": prompt,
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.post(self.base_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 429:
                            retry_after = float(resp.headers.get('Retry-After', 1))
                            print(f"  Rate limited on batch {batch_idx+1}, waiting {retry_after}s...", flush=True)
                            await asyncio.sleep(retry_after)
                            continue
                        
                        if resp.status != 200:
                            text = await resp.text()
                            print(f"  Error batch {batch_idx+1} (status {resp.status}): {text[:200]}", flush=True)
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return batch_idx, [False] * len(documents)
                        
                        data = await resp.json()
                        answer = data.get('output', [{}])
                        # Extract text from response
                        if isinstance(answer, list):
                            answer_text = ""
                            for block in answer:
                                if isinstance(block, dict) and block.get('type') == 'message':
                                    for content in block.get('content', []):
                                        if isinstance(content, dict) and content.get('type') == 'output_text':
                                            answer_text = content.get('text', '')
                                elif isinstance(block, dict) and block.get('text'):
                                    answer_text = block.get('text', '')
                        else:
                            answer_text = str(answer)
                        
                        answer_text = answer_text.strip().upper()
                        answers = [ans.strip() == "YES" for ans in answer_text.split(',')]
                        
                        # Pad or truncate to match document count
                        if len(answers) < len(documents):
                            answers.extend([False] * (len(documents) - len(answers)))
                        answers = answers[:len(documents)]
                        
                        return batch_idx, answers
                        
                except asyncio.TimeoutError:
                    print(f"  Timeout on batch {batch_idx+1}, attempt {attempt+1}/{max_retries}", flush=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue
                except Exception as e:
                    print(f"  Error on batch {batch_idx+1}: {e}", flush=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue
            
            return batch_idx, [False] * len(documents)

    async def filter_documents_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter documents using parallel async API calls"""
        query = data.get("question", "")
        documents = data.get("ctxs", [])
        
        if not query or not documents:
            return data
        
        print(f"  Query: {query[:100]}...", flush=True)
        print(f"  Evaluating {len(documents)} documents ({len(documents)//self.batch_size + 1} batches, {self.max_concurrent} concurrent)...", flush=True)
        
        # Split into batches
        batches = []
        for i in range(0, len(documents), self.batch_size):
            batches.append(documents[i:i + self.batch_size])
        
        # Run all batches concurrently with semaphore limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.verify_batch_async(session, query, batch, idx, semaphore)
                for idx, batch in enumerate(batches)
            ]
            results = await asyncio.gather(*tasks)
        
        # Sort results by batch index and flatten
        results.sort(key=lambda x: x[0])
        all_results = []
        for _, batch_results in results:
            all_results.extend(batch_results)
        
        # Filter gold documents
        gold_documents = [
            doc for doc, is_gold in zip(documents, all_results) if is_gold
        ]
        
        filtered_data = data.copy()
        filtered_data["ctxs"] = gold_documents
        
        # Print summary
        for i, (doc, is_gold) in enumerate(zip(documents, all_results)):
            title = doc.get('title', 'No title')[:50]
            status = "✓ GOLD" if is_gold else "✗ FILTERED"
            print(f"    {status} - Document {i+1}: {title}...", flush=True)
        
        print(f"  Original: {len(documents)} | Gold: {len(gold_documents)} | Filtered: {len(documents) - len(gold_documents)}", flush=True)
        
        return filtered_data

    def filter_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync wrapper for async filter"""
        return asyncio.run(self.filter_documents_async(data))


def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict[str, Any]], int]:
    """Load processed entries from checkpoint"""
    if not os.path.exists(checkpoint_file):
        return [], 0
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            processed_entries = checkpoint_data.get('processed_entries', [])
            last_line = checkpoint_data.get('last_processed_line', 0)
            print(f"Loaded checkpoint: {len(processed_entries)} entries, resuming from line {last_line + 1}", flush=True)
            return processed_entries, last_line
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting fresh.", flush=True)
        return [], 0


def save_checkpoint(checkpoint_file: str, processed_entries: List[Dict[str, Any]], last_line: int):
    """Save progress to checkpoint"""
    checkpoint_data = {
        'processed_entries': processed_entries,
        'last_processed_line': last_line,
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(processed_entries)
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    print(f"  Checkpoint saved: {len(processed_entries)} entries", flush=True)


async def process_multiple_entries_async(verifier: DocumentVerifier, input_file: str, output_file: str, 
                                          parallel_queries: int = 5):
    """Process multiple queries in parallel"""
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    processed_entries, start_line = load_checkpoint(checkpoint_file)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Total queries: {total_lines} | Already processed: {start_line} | Remaining: {total_lines - start_line}", flush=True)
    
    # Process queries in chunks for parallel execution
    remaining_lines = lines[start_line:]
    
    for chunk_start in range(0, len(remaining_lines), parallel_queries):
        chunk = remaining_lines[chunk_start:chunk_start + parallel_queries]
        chunk_data = []
        
        for line in chunk:
            try:
                chunk_data.append(json.loads(line))
            except json.JSONDecodeError:
                chunk_data.append(None)
        
        # Process chunk in parallel
        tasks = []
        for data in chunk_data:
            if data is not None:
                tasks.append(verifier.filter_documents_async(data))
            else:
                tasks.append(asyncio.coroutine(lambda: None)())
        
        actual_line = start_line + chunk_start
        print(f"\n[{actual_line + 1}-{actual_line + len(chunk)}/{total_lines}] Processing {len(chunk)} queries in parallel...", flush=True)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  Error on query {actual_line + i + 1}: {result}", flush=True)
                # Use original data on error
                if chunk_data[i] is not None:
                    processed_entries.append(chunk_data[i])
            elif result is not None:
                processed_entries.append(result)
        
        # Save checkpoint after each chunk
        save_checkpoint(checkpoint_file, processed_entries, start_line + chunk_start + len(chunk))
    
    # Save final output
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Checkpoint removed.", flush=True)
    
    return processed_entries


def main():
    parser = argparse.ArgumentParser(description='Parallel document verification using OpenAI API')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--batch-size', type=int, default=5, help='Documents per API call (default: 5)')
    parser.add_argument('--max-concurrent', type=int, default=50, help='Max concurrent API calls (default: 50)')
    parser.add_argument('--parallel-queries', type=int, default=5, help='Number of queries to process in parallel (default: 5)')
    parser.add_argument('--input-file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output-file', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--model', type=str, default='gpt-5-nano', help='Model to use (default: gpt-5-nano)')
    
    args = parser.parse_args()
    
    verifier = DocumentVerifier(
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size
    )
    
    print(f"Config: batch_size={args.batch_size}, max_concurrent={args.max_concurrent}, parallel_queries={args.parallel_queries}", flush=True)
    
    start_time = time.time()
    
    processed_entries = asyncio.run(
        process_multiple_entries_async(verifier, args.input_file, args.output_file, args.parallel_queries)
    )
    
    elapsed = time.time() - start_time
    print(f"\nDone! Processed {len(processed_entries)} entries in {elapsed:.1f}s ({elapsed/max(len(processed_entries),1):.2f}s/query)", flush=True)


if __name__ == "__main__":
    main()