import random
import torch
from pathlib import Path
import os
import logging
from tqdm import tqdm
import argparse
import json
import csv

import transformers
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def write_tsv(filename, data):
    with open(filename, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerows(data)

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

  


def main(args):
    random.seed(0)
    torch.manual_seed(0)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    
    # corpus_and_retrievers = [('sphere', 'bm25'), ('sphere', 'dpr'), ('sphere', 'contriever')]
    # corpus_and_retrievers = [('google_api', 'bm25'), ('google_api', 'contriever')]
    corpus_and_retrievers = [('wiki', 'bm25'), ('wiki', 'dpr'), ('wiki', 'contriever'), ('sphere', 'bm25'), ('sphere', 'dpr'), ('sphere', 'contriever'), ('google_api', 'bm25')]
    
    for data_type in ['arguana_generated', 'kialo', 'opinionqa']: # compute ensemble mrecall for each dataset
        
        for corpus, retriever in corpus_and_retrievers:
            
            # get original data
            file_path = Path(args.root) / Path(str(corpus)) / Path(str(retriever)) / 'raw_query_exp' / f'{data_type}_1k_query_exp.jsonl'
            logger.info(f'processing {file_path}')
            data = read_jsonl(file_path)
            
            # if data_type == 'arguana_generated':
            #     data = data[:750]
            # elif data_type == 'kialo':
            #     data = data[:774]
            # elif data_type == 'opinionqa':
            #     data = data[:882]
            # else:
            #     raise NotImplementedError
            
            


            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/scratch/cluster/hungting/projects/Multi_Answer/Data/retrieval_outputs")
    
    args = parser.parse_args()

    main(args)
    
    # PYTHONPATH=.. python test.py 
