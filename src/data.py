from datasets import load_dataset
import json
from tqdm import tqdm

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_data(dataset_name, retrieved_doc_type, split):
    if dataset_name == 'vk':
        assert split == 'test'
        import json 
        dataset = json.load(open('modular_pluralism/input/overton_test_valuekaleidoscope.json'))
    elif dataset_name == 'debateqa':
        dataset = read_jsonl('DebateQA/dataset/dev.jsonl')
    elif dataset_name in ['qampari', 'quest']:
        dataset = read_jsonl(f'data/{dataset_name}_data/dev_data.jsonl')
        if retrieved_doc_type == 'none':
            return dataset
        if retrieved_doc_type in ['bm25', 'nv', 'contriever']:
            retrieved_docs = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/{dataset_name}/{dataset_name}_{retrieved_doc_type}.jsonl')
        else:
            raise ValueError(f"Unknown retrieved_doc_type: {retrieved_doc_type}")
        
        for retrieved_doc, inst in tqdm(zip(retrieved_docs, dataset)):
            documents = retrieved_doc['ctxs']
            documents = [doc['title'] + ' ' + doc['text'] for doc in documents]
            inst['ctxs'] = documents
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset


def prepare_retrieval_prompts(question, docs, num_step, tokenizer, instruction, support_system_prompt=False):
    use_openai = (tokenizer == None)
    prompt, instruction = instruction.split('<END_SYSTEM>')[0].strip().strip('\n'), instruction.split('<END_SYSTEM>')[1].strip()
    content = instruction.replace("[Question]", question).replace("[Documents]", "\n".join(docs)).replace("[Counter]", num_step)
    # content = "Documents:\n[Documents]\n\nOriginal Question: [Question]\nRewritten Query:\n".replace("[Question]", question).replace("[Documents]", "\n".join(docs))

    # compose messages 
    if support_system_prompt:
        messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ]
    else:
        messages = [
                {"role": "user", "content": prompt + '\n\n' + content},
            ]

    if use_openai:
        return messages
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    

def prepare_rag_prompts(dataset, tokenizer, data_type, instruction, diverse=True, rag=False, support_system_prompt=False):
    use_openai = (tokenizer == None)
    if diverse:
        prompt = f"Make sure your response reflects diverse values and perspectives for the following instruction. {instruction}"
    else:
        prompt = instruction
        
    inputs = []
    
    question_str = 'question'
    if 'question_text' in dataset[0]:
        question_str = 'question_text'
    elif 'query' in dataset[0]:
        question_str = 'query'
        
    for example in dataset:
        # construct content
        if data_type == 'vk':
            content = example['input']
        else:
            if rag:
                content = "Documents:\n[Documents]\n\nQuestion: [Question]\nAnswer:\n".replace("[Question]", example[question_str]).replace("[Documents]", "\n".join(example['ctxs']))
            else:
                content = "Question: [Question]\nAnswer:".replace("[Question]", example[question_str])
        # compose messages 
        if support_system_prompt:
            messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ]
        else:
            messages = [
                    {"role": "user", "content": prompt + '\n\n' + content},
                ]
        
        if use_openai:
            inputs.append(messages)
        else:
            inputs.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return inputs



def get_instruction(instruction_file):
    instruction = open(instruction_file, "r").read().strip()
    return instruction