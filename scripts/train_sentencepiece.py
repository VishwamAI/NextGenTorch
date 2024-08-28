import os
import sentencepiece as spm
import yaml

# Load configurations
with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

def train_sentencepiece(model_type, input_file, model_prefix, vocab_size):
    """
    Train a SentencePiece tokenizer model for a specific model type.

    Args:
    model_type (str): The type of model (e.g., 'gpt', 'bert', 'roberta', 't5', 'albert')
    input_file (str): Path to the input text file for training
    model_prefix (str): Prefix for the output model files
    vocab_size (int): Size of the vocabulary to be created
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]'
    )
    print(f"SentencePiece model for {model_type} trained and saved with prefix: {model_prefix}")

def main():
    # Specify the input file path (you need to create this file with your training data)
    input_file = '/home/ubuntu/nextgentorch/data/tokenizer_training_data.txt'

    # Ensure the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Train SentencePiece models for each model type
    for model_type in ['gpt', 'bert', 'roberta', 't5', 'albert']:
        vocab_size = model_config[model_type][list(model_config[model_type].keys())[0]]['vocab_size']
        model_prefix = f'/home/ubuntu/nextgentorch/models/tokenizers/spiece_{model_type}'
        train_sentencepiece(model_type, input_file, model_prefix, vocab_size)

if __name__ == "__main__":
    main()
