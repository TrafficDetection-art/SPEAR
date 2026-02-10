import sys; sys.path.insert(0, '..')
from project_settings import get

import json
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def json_to_string(data, indent=0):
    result = []
    indent_str = '  ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result.append(f'{indent_str}{key}:')
                result.append(json_to_string(value, indent + 1))
            else:
                result.append(f'{indent_str}{key}: {value}')
    elif isinstance(data, list):
        for item in data:
            result.append(json_to_string(item, indent))
    else:
        result.append(f'{indent_str}{data}')
    return '\n'.join(result)


def tokenize_function(examples, max_length=None):
    if max_length is None:
        max_length = get("dl.model.max_len", 512)
    tokenizer_name = get("dl.tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer(examples['Text'], padding="max_length", truncation=True, max_length=max_length)


def load_dataset(data_path, tokenizer, max_length=None):
    if max_length is None:
        max_length = get("dl.model.max_len", 512)
    with open(data_path, "r") as f:
        json_data = json.load(f)
    
    # Filter out entries with missing 'Class' values
    filtered_data = [item for item in json_data if "Class" in item]
    
    if len(filtered_data) < len(json_data):
        print(f"Warning: Skipped {len(json_data) - len(filtered_data)} entries with missing 'Class' values")
    
    formatted_data = {
        "Text": [str(item["Text"]) if isinstance(item["Text"], str) else "" for item in filtered_data],
        "label": [item["Class"] for item in filtered_data],
        "type": [item.get("type", "unknown") for item in filtered_data],
    }
    dataset = Dataset.from_dict(formatted_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
