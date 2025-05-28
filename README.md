# Training a Diverse Retriever

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

## Data

## Checkpoints