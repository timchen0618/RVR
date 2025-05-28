import random
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import ast

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_jsonl(file_path):
    import json
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

class Inferencer:
    def __init__(self, model_id, n_gpu, instruction):
        self.model_id = model_id
        self.llm = LLM(model=model_id,
                    tokenizer=self.model_id,
                    enable_prefix_caching=True,
                    trust_remote_code=True,
                    tensor_parallel_size=n_gpu,
                    gpu_memory_utilization=0.9)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=16, seed=0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.instruction = instruction
    
    def formulate_dialogs(self, doc_1, doc_2):
        start_tok = '<|im_start|>'
        end_tok = '<|im_end|>'
        _instruction = self.instruction.replace('[Doc 1]', doc_1).replace('[Doc 2]', doc_2)
        messages = _instruction.split(end_tok)[:-1]
        messages = [{"role": m.strip('\n').split('\n')[0][12:], "content": '\n'.join(m.strip('\n').split('\n')[1:]).strip()} for m in messages]    
        return messages
    
    def prepare_input(self, data, answers):
        input_texts = []  # num_instance * num_perspectives * num_docs
        for inst, answer in tqdm(zip(data, answers)):
            # retrieving perspectives
            perspectives = inst['perspectives']
            
            # for every perspective, check if it is supported by the answer
            for p in perspectives:
                input_texts.append(self.formulate_dialogs(answer['answer'], p)[1:])
        return input_texts
                        
        
    def inference(self, input_texts):
        self.llm.generate(input_texts[0], self.sampling_params)
    
        outputs = self.llm.generate(input_texts, self.sampling_params, use_tqdm=True)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
    
    def parse_response(self, response):
        if response[0] != '{':
            response = response.strip().strip('\n').split('Answer:')[-1].split('The answer is')[-1].split('The answer is \"')[-1].split('The answer is:')[-1].split('Based on the information provided in the document, the answer is')[-1].split('Based on the information provided in the document, the answer is:\n')[-1].strip()
            response = response.strip().lower()
            _yes = (response[:3] == 'yes' or response[:4] == '(yes') or (response == 'y' or response == 'ye') or (response[-3:] == 'yes' or response[-4:] == 'yes.' or response[-5:] == 'yes\".')
            _entail_or_true = (response[:6] == 'entail' or (response[:4] == 'true'))
            return  _yes or _entail_or_true
        else:
            return ast.literal_eval(response)['Answer'].strip().lower() == 'yes'



def prepare_predictor_and_inputs(args, data, answers):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # read instructions
    instruction = open(args.instructions).read()
    inferencer = Inferencer(args.model, 1, instruction)
    
    input_texts = inferencer.prepare_input(data, answers)
    input_texts = inferencer.tokenizer.apply_chat_template(input_texts, add_generation_prompt=True, tokenize=False)
    return inferencer, input_texts
    

def main(args):
    random.seed(0)
    torch.manual_seed(0)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    
    preds = []
    
    logger.info('running perspective detection: eval')
    
    # prepare data
    data = load_dataset(args.data, split='test')
    answers = read_jsonl(args.answers)
    
    # prepare predictor 
    logger.info('loading model and data...')
    inferencer, input_texts = prepare_predictor_and_inputs(args, data, answers)
    logger.info('finish loading model, start inference...')
    outputs = inferencer.inference(input_texts)
    # print(outputs)
    outputs = [inferencer.parse_response(output) for output in outputs]
    # print(outputs)
    
    
    for inst in tqdm(data):
        pred_inst = []  # (num_perspectives)
        # for every perspective, check if it is supported by the docs
        for p in inst['perspectives']:
            pred_inst.append(outputs.pop(0))

        preds.append(pred_inst)
    # Check if each item in preds is all True
    completeness = [all(pred) for pred in preds]
    percentage = sum(completeness) / len(completeness) * 100
    
    # Flatten the preds list
    flattened_preds = [pred for sublist in preds for pred in sublist]
    # Compute the percentage of True values
    true_percentage = sum(flattened_preds) / len(flattened_preds) * 100
    
    print(f"Percentage of perspectives that are supported: {true_percentage:.2f}%")
    print(f"Percentage of instances where all perspectives are supported: {percentage:.2f}%")
        
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="timchen0618/Mistral_BERDS_evaluator_full", type=str)
    parser.add_argument("--model_type", default="mistral", type=str)
    parser.add_argument("--instructions", type=str, default="../berds/Eval/instructions_chat.txt")
    parser.add_argument("--answers", type=str, default="outputs/Meta-Llama-3.1-8B-Instruct_OpinionQA_diverse.jsonl")    
    parser.add_argument("--data", type=str, default="timchen0618/OpinionQA")
    args = parser.parse_args()

    main(args)
    
    # python eval.py --answers  