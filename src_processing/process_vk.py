import json

def process_vk_dataset(input_file, output_file, new_file):
    # Read input file
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # Read output file
    with open(output_file, 'r') as f:
        output_data = [json.loads(line) for line in f]

    # Add "answer" from output to "output" field in input
    for input_item, output_item in zip(input_data, output_data):
        input_item['output'] = output_item['answer']

    # Write the updated input to a new file
    with open(new_file, 'w') as f:
        json.dump(input_data, f, indent=2)

# Example usage
input_file = 'modular_pluralism/input/overton_test_valuekaleidoscope.json'
models = ["gemma-2-9b-it", "Meta-Llama-3.1-8B-Instruct"]
# diverse_options = [True, False]
diverse_options = [False]

for model in models:
    for diverse in diverse_options:
        diverse_suffix = "_diverse" if diverse else ""
        output_file = f'{model}_vk{diverse_suffix}.jsonl'
        new_file = f'processed_{output_file}'[:-1]

        process_vk_dataset(input_file, 'outputs/' + output_file, 'modular_pluralism/output/' + new_file)
        print(f"Processed data has been written to {new_file}")
