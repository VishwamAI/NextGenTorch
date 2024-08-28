import os
import json
import re
from typing import List, Tuple, Dict
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerFast, SentencePieceTokenizer
import sentencepiece as spm
from sympy import sympify, SympifyError
from sklearn.model_selection import train_test_split
import numpy as np
import torch

# Load configurations
with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/dataset_paths.yaml', 'r') as f:
    dataset_paths = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
    training_config = yaml.safe_load(f)

def get_tokenizer(model_type: str):
    if model_type in ['gpt', 'bert', 'roberta', 't5', 'albert']:
        tokenizer = SentencePieceTokenizer.from_pretrained(f"spiece_{model_type}.model")
        if model_type == 'gpt':
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Initialize tokenizer based on the selected model type
model_type = training_config['model']['type'].lower()
model_size = training_config['model']['size']
tokenizer = get_tokenizer(model_type)

# Get max_length from model config
max_length = model_config[model_type][model_size]['hidden_size']

def load_dataset(path: str) -> List[dict]:
    """Load dataset from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Normalize LaTeX-like notation
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    # Remove any remaining extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_math(expression: str) -> List[str]:
    """Tokenize mathematical expressions."""
    try:
        # Use SymPy to parse and tokenize mathematical expressions
        parsed = sympify(expression)
        return [str(symbol) for symbol in parsed.free_symbols]
    except SympifyError:
        # If SymPy fails, fall back to simple splitting
        return expression.split()

def pad_sequence(sequence: List[int], max_len: int, pad_token_id: int) -> List[int]:
    """Pad or truncate a sequence to max_len."""
    if len(sequence) > max_len:
        return sequence[:max_len]
    return sequence + [pad_token_id] * (max_len - len(sequence))

def preprocess_example(example: dict, model_type: str, tokenizer) -> Dict[str, torch.Tensor]:
    """Preprocess a single example from the dataset using SentencePiece tokenization."""
    question = clean_text(example['question'])
    answer = clean_text(example['answer'])

    # Tokenize text using SentencePiece for all model types
    if model_type in ['gpt', 'bert', 'roberta', 'albert']:
        encoded = tokenizer.encode_plus(
            question,
            text_pair=answer,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }
    elif model_type == 't5':
        question_encoded = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        answer_encoded = tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': question_encoded['input_ids'].squeeze(),
            'attention_mask': question_encoded['attention_mask'].squeeze(),
            'decoder_input_ids': answer_encoded['input_ids'].squeeze(),
            'decoder_attention_mask': answer_encoded['attention_mask'].squeeze(),
            'labels': answer_encoded['input_ids'].squeeze()
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def preprocess_dataset(dataset: List[dict], tokenizer, model_type: str) -> List[Dict[str, torch.Tensor]]:
    """Preprocess entire dataset."""
    return [preprocess_example(example, model_type, tokenizer) for example in dataset]

def split_dataset(data: List[Tuple[torch.Tensor, torch.Tensor]], train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Split the dataset into training, validation, and test sets."""
    train_val, test = train_test_split(data, test_size=test_size, random_state=random_state)
    relative_val_size = val_size / (train_size + val_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, random_state=random_state)

    return {
        'train': train,
        'validation': val,
        'test': test
    }

def save_split_datasets(split_data: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]], base_path: str):
    """Save the split datasets."""
    for split_name, data in split_data.items():
        file_path = os.path.join(base_path, f'{split_name}.pt')
        torch.save(data, file_path)
        print(f"Saved {split_name} set to {file_path}")

def main():
    # Load configurations
    with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)

    model_type = training_config['model']['type'].lower()
    model_size = training_config['model']['size']

    # Initialize tokenizer
    tokenizer = get_tokenizer(model_type, model_size)

    # Load datasets
    gsm8k_path = dataset_paths['gsm8k']['path']
    math_path = dataset_paths['math']['path']

    gsm8k_data = load_dataset(gsm8k_path)
    math_data = load_dataset(math_path)

    # Preprocess datasets
    gsm8k_processed = preprocess_dataset(gsm8k_data, tokenizer, model_type)
    math_processed = preprocess_dataset(math_data, tokenizer, model_type)

    # Split datasets
    gsm8k_split = split_dataset(gsm8k_processed)
    math_split = split_dataset(math_processed)

    # Save split datasets
    save_split_datasets(gsm8k_split, dataset_paths['output']['gsm8k'])
    save_split_datasets(math_split, dataset_paths['output']['math'])

    print(f"Data preprocessing and splitting completed for {model_type.upper()} {model_size} model.")

if __name__ == "__main__":
    main()
