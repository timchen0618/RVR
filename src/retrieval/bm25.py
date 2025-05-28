import numpy as np
from typing import List, Union
from rank_bm25 import BM25Okapi
from nltk import word_tokenize
import csv
from tqdm import tqdm
import json

from src.retrieval.base_retriever import BaseRetriever


import sys
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        


def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(filename, data):
    with open(filename, 'a') as f:
        for inst in data:
            f.write(json.dumps(inst) + '\n')



        



from pyserini.search.lucene import LuceneSearcher

class BM25LuceneSearcher(BaseRetriever):
    def __init__(self, corpus_path: str) -> None:
        super().__init__()
        # initialize retriever
        self.searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')

    def retrieve(self, query: str, n: int) -> List[str]:
        hits = self.searcher.search(query, k=n)
        docs = []
        for i in range(0, n):
            # print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
            try:
                doc = self.searcher.doc(hits[i].docid)
            except IndexError:
                print(f'IndexError: {i}')
                continue
            json_doc = json.loads(doc.raw())
            title = json_doc['contents'].split('\n')[0].strip("\"")
            text = '\n'.join(json_doc['contents'].split('\n')[1:])
            docs.append({'title': title, 'text': text, 'score': hits[i].score})
        return docs
    
    def build_corpus(self, corpus_path: str) -> None:
        pass


class BM25Retriever(BaseRetriever):
    def __init__(self, corpus_path: str) -> None:
        super().__init__()
        # build corpus
        self.build_corpus(corpus_path)

        # initialize retriever
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, n: int) -> List[str]:
        tq = word_tokenize(query)
        docs = self.bm25.get_top_n(tq, self.tokenized_corpus, n=n)
        docs = [{'title': self.corpus[self.tokenized2id[' '.join(doc)]].split('[SEP]')[0], 'text': self.corpus[self.tokenized2id[' '.join(doc)]].split('[SEP]')[1]} for doc in docs]
        docs = [doc['title'] + ' ' + doc['text'] for doc in docs]
        return docs
    
    def read_corpus(self, file_path: str) -> List[str]:
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                data.append(row[2] + '[SEP]' + row[1])
        return data
    
    def build_corpus(self, corpus_path: str) -> None:
        self.corpus = self.read_corpus(corpus_path)
        tokenized_corpus, tokenized2id = self.tokenize_corpus()
        self.tokenized_corpus = tokenized_corpus
        self.tokenized2id = tokenized2id
    
    def tokenize_corpus(self) -> None:
        # tokenize the corpus
        tokenized2id = {}
        tokenized_corpus = []
        for _id, doc in enumerate(self.corpus):
            tokenized_doc = word_tokenize(doc)
            tokenized_corpus.append(tokenized_doc)
            tokenized2id[' '.join(tokenized_doc)] = _id
        return tokenized_corpus, tokenized2id
    