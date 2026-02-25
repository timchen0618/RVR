# Training a Diverse Retriever
## Requirements
A `requirements.txt` is included for packages to install.  

### Using Python Virtualenv
First create a virtual environment via 
`python3 -m venv diverse_retrieval`
Then activate the virtual environment
`source activate diverse_retrieval/bin/activate`


## Codebase Structure
- `training/` (see README.md in training/)
- `eval/`:
    - `eval_qampari.py`: for evaluating retrieval results.
    - `statistical_significance.py`: for evaluating statstical significance of results
    - `verifier_performance.py`: for evaluating performance of verifiers
- `baselines/`:
    - `query_exp`: implement query expansion using GPT models. 
    - `reranking`: implement some existing rerankers.
    - `utils`: utility functions.

- `data/`: you should get the data following the [Data](#data) section. 

- `iterative_retrieval.py`: baseline which does iterative retrieval. 
- `web_demos/`: runs the app that exports retrieval result to an external website (for observation).
- `src/`
    - `eval/`: evaluation scripts.
    - `retrieval/`: define retriever classes.
    - `utils/`: utility functions.
- `src_processing/`: get_chunk.py ==> get wikipedia chunks from pages.



## Running Retrieval (Inference)
### Getting the Corpus
1. Download Wikipedia chunks from [QAMPARI_Website](https://samsam3232.github.io/qampari/). Click on "Download chunked wikipedia dump" and save the file to your machine. 
2. Unzip the downloaded file you should see the `wikipedia_chunks/chunks_v5/` folder. 
3. Each file in the `wikipedia_chunks/chunks_v5/` folder is a JSONL file, where each line contains one document chunk. 
4. Parse the JSONL files to into a single `.tsv` file, with the format of `[id]\t[text]\t[title]`. 
5. If you load the data to variable `data`, then the id of each element is `data[i]['id']`. Title: `data[i]['meta']['title']`. Text: `data[i]['contents']`. 
6. Name the `.tsv` file to be `wikipedia_chunks/chunks_v5.tsv`.


### Generating the Document Indices
Follow the scripts in `inference/gen_embeddings/{model_name}/gen_embed.sbatch`. 

### Retrieve-Verify-Retrieve (RVR)

The pipeline alternates between retrieval and verification:

**Round 1 Retrieve → Verify → Round 2 Retrieve → Verify → ...**

#### Retrieve

Script: `inference/{dataset}/round_t_retrieve.sbatch`

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Path to your model |
| `passages` | Path to your corpus |
| `passages_embeddings` | Path to your passage embeddings |
| `data` | Path to data file. For round 1, our setup does not use input documents. For subsequent rounds, the Verify step generates a file with input documents |
| `output_dir` | Path to output directory |
| `outfile` | Desired name of output file (e.g. `round2_retrieved.json`) |
| `training_mode` | Should match `training_mode` used in `finetuning.py` |
| `n_docs` | Number of docs to retrieve |

#### Verify

Script: `inference/{dataset}/round_t_verify.sbatch`

| Parameter | Description |
|-----------|-------------|
| `initial_retrieval_file` | Retrieved file from the previous round |
| `source_queries_file` | Path to ground truth file (e.g. `data/dev_data_gt_webqsp.jsonl`) |
| `output_base_dir` | Path to output directory |
| `top_n_concat` | Context budget (number of documents to use as input) |
| `round_num` | If > 1, generates both the evaluation file for this round and the input file for the next round's retrieval |
| `verifier` | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `batch_size` | `16` |
| `verify_top_k` | `50` (verifier budget B = 50) |

### Evaluation
Run the last cell of `eval/eval_qampari.py`.   

1. Change the `''` in `file_list` to be the files containing the retrieved documents. 
2. Change the `gt_data_path` to be the ground truth data (QAMPARI, QUEST, WebQSP). 

**Ground Truth Data**
1. For settings where the model is conditioned on some ground truth documents as input data (the model is doing only the 2nd stage retrieval), use data in `data/qampari_data/2nd_stage_test_data`.  
2. For settings where the model does not have access to the ground truth documents, use `data/qampari_data/dev_data_gt_qampari_corpus.jsonl` as the ground truth.   

**Metrics**
*has_gold_id=True will only work for QAMPARI as we do not have ground truth documents for QUEST or WebQSP*
1. Set `has_gold_id=False` to use "Answer Recall", which computes recall by looking at each document and see if it contains an answer. (A more lenient metric) 
2. Set `has_gold_id=True` to use "Evidence Recall", which computes recall by comparing the retrieved documents against "ground truth" documents. (A stricter metric)


## Data
### Train with Random Negatives (Default)

##### Train with Three Documents 
To fine-tune the round 1 (D_i) model with 0 input documents, use file `train_random+input_data.jsonl`. Each element is a `JSON` file. The data structure is a follows. Ensure `training_mode = base`:  
```
[
    {
        'id': id,
        'question': "Question:[Question]\n\n\Documents:[Documents]",
        'org_q': "[Question]",
        'positive_ctxs': [ground documents minus the input negative documents], 
        'negative_ctxs': [random negative documents], 
        'hard_negative_ctxs': [hard negatives; this list should be empty], 
        'input_negative_ctxs': [the input document list. length should be three]
    }
]
```
##### Train with Varying Number of Documents
Use `training/data/qampari_data/qampari_corpus/train_data.jsonl` instead. The data structure is as follows: 
```
[
    {
        'qid': id,
        'question_text': "[Question]",
        'positive_ctxs': [ground documents], 
        'negative_ctxs': [random negative documents], 
        'hard_negative_ctxs': [hard negatives; this list should be empty],
        'answer_list': list_of_answers
    }
]
```

Specify `--sample_length --doc_length [number_list]` to use the `SampleDataset` class in `finetuning_data.py`.  
To fetch a new batch, the script will first sample a number from `[number_list]`, and then sample that number of positive documents from the pool as input documents. The remaining positive documents are the actual "ground truth". Specify `--norm_doc --norm_query` for Qwen3 and INF models. Ensure `training_mode = standard`

### Train with Hard Negatives 
**not done in our paper**
##### Mine Hard Negatives from Model Outputs
1. Retrieve top 100 documents for the training set (`train_random+input_data.jsonl`). In the case of varying number of input documents, use `training/data/qampari_data/qampari_corpus/train_data.jsonl`.
2. Compare the top 100 with the gold document set (evidence recall against inp0, since it contains all the positive documents) and see if they contain the answer (answer recall, gt_qampari), making sure to remove all the gold documents. For `training/data/qampari_data/qampari_corpus/train_data.jsonl` it already has all the gold documents so for evidecne recall only compare to its `positive_ctxs` field. 
3. Take the top negative documents as the hard negative and add to `hard_negative_ctxs` in the training file and rerun the training process. 


