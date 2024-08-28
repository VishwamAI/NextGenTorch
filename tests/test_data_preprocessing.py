import pytest
import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sentencepiece import SentencePieceProcessor
from scripts.data_preprocessing import (
    get_tokenizer,
    preprocess_example,
    preprocess_dataset,
    clean_text,
    tokenize_math,
    pad_sequence,
    SentencePieceTokenizer,
)

# Load configurations
with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
    training_config = yaml.safe_load(f)

@pytest.fixture(params=['gpt', 'bert', 'roberta', 't5', 'albert'])
def model_type(request):
    return request.param

@pytest.fixture
def mock_example():
    return {
        'question': 'What is 2 + 2?',
        'answer': 'The answer is 4.'
    }

def test_get_tokenizer(model_type):
    tokenizer = get_tokenizer(model_type)
    assert tokenizer is not None
    assert isinstance(tokenizer, SentencePieceTokenizer)
    if model_type == 'gpt':
        assert tokenizer.pad_token == tokenizer.eos_token
    # Additional assertions can be added here if needed for specific model types

def test_preprocess_example(model_type, mock_example):
    tokenizer = get_tokenizer(model_type)
    processed = preprocess_example(mock_example, model_type, tokenizer)

    assert isinstance(processed, dict)
    assert 'input_ids' in processed
    assert 'attention_mask' in processed
    assert isinstance(processed['input_ids'], torch.Tensor)
    assert isinstance(processed['attention_mask'], torch.Tensor)

    if model_type == 't5':
        assert 'decoder_input_ids' in processed
        assert 'decoder_attention_mask' in processed
        assert isinstance(processed['decoder_input_ids'], torch.Tensor)
        assert isinstance(processed['decoder_attention_mask'], torch.Tensor)

def test_preprocess_dataset(model_type):
    mock_dataset = [
        {'question': 'What is 2 + 2?', 'answer': 'The answer is 4.'},
        {'question': 'What is the capital of France?', 'answer': 'The capital of France is Paris.'}
    ]
    tokenizer = get_tokenizer(model_type)
    processed = preprocess_dataset(mock_dataset, tokenizer, model_type)

    assert isinstance(processed, list)
    assert len(processed) == len(mock_dataset)
    for item in processed:
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)

def test_clean_text():
    text = "  This is a   test  with $x^2$ and $$y = mx + b$$ equations.  "
    cleaned = clean_text(text)
    expected = "This is a test with x^2 and y = mx + b equations."
    print(f"Actual cleaned text: '{cleaned}'")  # Debug print
    print(f"Expected cleaned text: '{expected}'")  # Debug print
    assert cleaned == expected, f"Expected: '{expected}', but got: '{cleaned}'"

def test_tokenize_math():
    expression = "x + 2*y - z^2"
    tokens = tokenize_math(expression)
    assert set(tokens) == {'x', 'y', 'z'}

def test_pad_sequence():
    sequence = [1, 2, 3, 4, 5]
    padded = pad_sequence(sequence, max_len=10, pad_token_id=0)
    assert len(padded) == 10
    assert padded == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]

    long_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    truncated = pad_sequence(long_sequence, max_len=10, pad_token_id=0)
    assert len(truncated) == 10
    assert truncated == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":
    pytest.main()
