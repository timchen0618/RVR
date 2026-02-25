import json
import statistics
from collections import Counter
from pathlib import Path

def analyze_quest_dataset(file_paths):
    """
    Analyze QUEST-format JSONL dataset files.
    
    Args:
        file_paths: List of paths to JSONL files (e.g., train.jsonl, test.jsonl)
    """
    
    all_data = []
    file_stats = {}
    
    # Load all data
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            all_data.extend(data)
            file_stats[Path(file_path).name] = len(data)
    
    # Basic counts
    num_queries = len(all_data)
    num_answers_per_query = [len(item['answer_list']) for item in all_data]
    
    # Question text analysis
    question_lengths = [len(item['question_text'].split()) for item in all_data]
    
    # Answer analysis
    all_answer_texts = []
    unique_answers = set()
    
    for item in all_data:
        for answer in item['answer_list']:
            answer_text = answer['answer_text']
            all_answer_texts.append(answer_text)
            unique_answers.add(answer_text)
    
    # Distribution analysis
    answer_count_distribution = Counter(num_answers_per_query)
    
    # Print statistics
    print("=" * 80)
    print("QUEST DATASET ANALYSIS")
    print("=" * 80)
    
    print("\n📊 DATASET OVERVIEW")
    print("-" * 80)
    for file_name, count in file_stats.items():
        print(f"  {file_name:20s}: {count:,} queries")
    print(f"  {'Total':20s}: {num_queries:,} queries")
    
    print("\n📝 QUERY STATISTICS")
    print("-" * 80)
    print(f"  Avg. query length:          {statistics.mean(question_lengths):.2f} words")
    print(f"  Median query length:        {statistics.median(question_lengths):.0f} words")
    print(f"  Min/Max query length:       {min(question_lengths)} / {max(question_lengths)} words")
    
    print("\n🎯 ANSWER STATISTICS")
    print("-" * 80)
    print(f"  Total answer instances:     {len(all_answer_texts):,}")
    print(f"  Unique answer entities:     {len(unique_answers):,}")
    print(f"  Avg. answers per query:     {statistics.mean(num_answers_per_query):.2f}")
    print(f"  Median answers per query:   {statistics.median(num_answers_per_query):.0f}")
    print(f"  Min/Max answers per query:  {min(num_answers_per_query)} / {max(num_answers_per_query)}")
    
    
    
    print("\n" + "=" * 80)
    
    # Return summary
    return {
        'num_queries': num_queries,
        'total_answers': len(all_answer_texts),
        'unique_answers': len(unique_answers),
        'avg_answers_per_query': statistics.mean(num_answers_per_query),
        'avg_query_length': statistics.mean(question_lengths),
        'file_stats': file_stats
    }


if __name__ == "__main__":
    # Modify these paths to your actual files
    file_paths = [
       # "dev_data_gt_quest.jsonl",
       #"dev_data_gt_webqsp.jsonl"
       'dev_data_gt_qampari_corpus.jsonl'
    ]
    
    # Or analyze a single file:
    # file_paths = ["your_file.jsonl"]
    
    stats = analyze_quest_dataset(file_paths)
    
    # Optionally print summary
    print("\n📋 SUMMARY")
    print(f"Total queries: {stats['num_queries']:,}")
    print(f"Total answer instances: {stats['total_answers']:,}")
    print(f"Unique entities: {stats['unique_answers']:,}")
    print(f"Avg answers/query: {stats['avg_answers_per_query']:.2f}")
