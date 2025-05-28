import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

from src.data import load_data, prepare_rag_prompts, get_instruction
from src.utils.schema import AnswerListSchema 

from outlines.serve.vllm import JSONLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run batched inference using vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset name from Hugging Face")
    parser.add_argument("--split", type=str, default="validation[:100]", help="Dataset split")
    parser.add_argument("--instruction_file", type=str, default="instruction_qa.txt", help="File containing the instruction for the model")
    parser.add_argument("--support_system_prompt", action="store_true", help="Support system prompt")
    parser.add_argument("--use_rag", action="store_true", help="Use RAG prompt")
    parser.add_argument("--retrieved_doc_type", type=str, default='none', help="Type of retrieved documents")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")

    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--use_outline", action='store_true', help='Use outline')

    return parser.parse_args()


def generate_openai_response(client, model, messages, seed, top_p, temperature, schema):
    if not schema:
        response = client.chat.completions.create(model=model, messages=messages, seed=seed, top_p=top_p, temperature=temperature)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_schema",
                    "schema": json.loads(schema)
                }
            },
            seed=seed,
            top_p=top_p,
            temperature=temperature
        )
        

    return response.choices[0].message.content


def parse_answers(answers):
    try:
        answers = json.loads(answers)['answers']
    except json.decoder.JSONDecodeError:
        print('answer', answers)
        answers = []
        print('Error')
    return answers


def run_inference(model, inputs, args, schema=None, use_tqdm=True):
    if 'gpt' in args.model_path:
        schema = schema.model_json_schema()
        outputs = [generate_openai_response(model, args.model_path, messages, seed=args.seed, top_p=args.top_p, temperature=args.temperature, schema=schema) for messages in tqdm(inputs)]
    else:
        # Set sampling parameters
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
        # Run vllm inference
        if schema:
            # outline generation
            assert args.use_outline
            sampling_params.logits_processors = [JSONLogitsProcessor(schema=schema, llm=model.llm_engine)]

        outputs = model.generate(inputs, sampling_params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text.strip() for output in outputs]
    return outputs 

def load_lm(args):
    if 'gpt' in args.model_path:
        from openai import OpenAI
        client = OpenAI()
        tokenizer = None
        return client, tokenizer
    else:
        llm = LLM(model=args.model_path,
                tokenizer=args.model_path,
                enable_prefix_caching=True,
                trust_remote_code=True,
                tensor_parallel_size=args.n_gpu,
                gpu_memory_utilization=0.9,
                disable_sliding_window=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        return llm, tokenizer
    
    
def main():
    args = parse_args()
    # Load model
    model, tokenizer = load_lm(args)

    # Load dataset
    assert args.use_rag == (args.retrieved_doc_type != 'none')
    dataset = load_data(args.dataset, args.retrieved_doc_type, args.split)

    # load instruction
    instruction_template = get_instruction(args.instruction_file)
    

    # Prepare prompts
    data_type = args.dataset
    inputs = prepare_rag_prompts(dataset, tokenizer, data_type, instruction_template, diverse=False, rag=args.use_rag, support_system_prompt=args.support_system_prompt)
    print(inputs[0])
    
    
    # Run inference
    if 'gpt' in args.model_path:
        outputs = run_inference(model, inputs, args, schema=AnswerListSchema)
        outputs = [parse_answers(output) for output in outputs]
    else:
        if args.use_outline:
            schema = AnswerListSchema
        else:
            schema = None
        outputs = run_inference(model, inputs, args, schema=schema)
        print(outputs[0])
        if args.use_outline:
            outputs = [parse_answers(output) for output in outputs]

    
    ###############################
    # Process and print results
    ###############################
    import json
    with open(f'outputs/{data_type.upper()}/{args.model_path.split("/")[-1]}_{data_type.split("/")[-1]}_r{args.retrieved_doc_type}.jsonl', 'w') as f:
        for i, output in enumerate(outputs):
            if data_type == 'debateqa':
                result = {"generation": output}
            else:
                result = {"answer": output}
            if "id" in dataset[i]:
                result["id"] = dataset[i]["id"]
            
            if "question" in dataset[i]:
                result["question"] = dataset[i]["question"]
            elif "question_text" in dataset[i]:
                result["question"] = dataset[i]["question_text"]
            elif "query" in dataset[i]:
                result["question"] = dataset[i]["query"]
            elif "input" in dataset[i]:
                result["input"] = dataset[i]["input"]
            else:
                raise ValueError(f"Unknown data type: {dataset[i]}")
            
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()

