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
- `eval_qampari.ipynb`: for evaluating retrieval results. 
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
Follow the script in `training/contriever/gen_embed.sh`. 

### Passage Retrieval
Follow the script in `training/contriever/retrieve.sh`. 

### Evaluation
Run the last cell of `eval_qampari.ipynb`.   

1. Change the `''` in `file_list` to be the files containing the retrieved documents. 
2. Change the `gt_data_path` to be the ground truth data. 

**Ground Truth Data**
1. For settings where the model is conditioned on some ground truth documents as input data (the model is doing only the 2nd stage retrieval), use data in `data/qampari_data/2nd_stage_test_data`.  
2. For settings where the model does not have access to the ground truth documents, use `data/qampari_data/dev_data_gt_qampari_corpus.jsonl` as the ground truth.   
3. Only flag `evaluate_repeat=True` if we're evaluating using the first setting (2nd stage setting).  

**Metrics**
1. Set `has_gold_id=False` to use "Answer Recall", which computes recall by looking at each document and see if it contains an answer. (A more lenient metric) 
2. Set `has_gold_id=True` to use "Evidence Recall", which computes recall by comparing the retrieved documents against "ground truth" documents. (A stricter metric)


## Data

## Checkpoints
