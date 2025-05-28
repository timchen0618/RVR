import json

def write_jsonl(data, file_path):
    """
    Write a list of dictionaries to a JSONL file.

    :param data: List of dictionaries to write to the file
    :param file_path: Path to the JSONL file
    """
    with open(file_path, 'w') as file:
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of dictionaries.

    :param file_path: Path to the JSONL file
    :return: List of dictionaries
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
import csv
def read_tsv(file_path):
    with open(file_path, 'r') as file:
        return [line for line in csv.reader(file, delimiter='\t')]
    
def write_tsv(data, file_path):
    with open(file_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(data)