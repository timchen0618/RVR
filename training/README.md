# Training (Contriever)
## 


## Codebase Structure

### Training `contriever/`
- 


### Data Processing
- `process_qampari_data.py` & `process_qampari_data.ipynb`: some processing function for making the QAMPARI data. Find the gold documents (that provide each answer to the question) within the Wikipedia corpus. 
- `get_training_data_qampari.ipynb`: obtain data for training diverse retrievers on QAMPARI from the matched data we got using the previous script. This would include "rewritten question", "positive documents", and "negative documents". 
- `get_data_quest.py`: obtain data for training diverse retrievers on QUEST. 

### Probing Answer Length (Predict number of answers)
- `probing/`: probing to see if we could predict answer length (number of answers) based on the question alone. 
