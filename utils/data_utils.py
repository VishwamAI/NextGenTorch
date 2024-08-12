import torch
from nextgenjax import tokenization
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim.oss import OSS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def preprocess_text(text: str) -> torch.Tensor:
    """
    Preprocess and tokenize input text using NextGenJax tokenization.

    Args:
        text (str): Input text to be preprocessed and tokenized.

    Returns:
        torch.Tensor: Tokenized and preprocessed text as a tensor.
    """
    # Use NextGenJax tokenization
    tokenizer = tokenization.get_tokenizer()
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens)

def create_data_loader(dataset, batch_size: int, shuffle: bool = True, distributed: bool = False):
    """
    Create a PyTorch DataLoader for the given dataset, with support for distributed training.

    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size for the data loader
        shuffle (bool): Whether to shuffle the data (default: True)
        distributed (bool): Whether to use DistributedSampler for distributed training (default: False)

    Returns:
        torch.utils.data.DataLoader: DataLoader for the given dataset
    """
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def pad_sequences(sequences: list, pad_value: int = 0) -> torch.Tensor:
    """
    Pad a list of sequences to the same length.

    Args:
        sequences (list): List of sequences to pad
        pad_value (int): Value to use for padding (default: 0)

    Returns:
        torch.Tensor: Padded sequences as a tensor
    """
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded_seqs)

def setup_fairscale(model, optimizer):
    """
    Set up Fairscale for distributed training.

    Args:
        model: The PyTorch model
        optimizer: The optimizer

    Returns:
        tuple: (ShardedDataParallel model, OSS optimizer)
    """
    model = ShardedDataParallel(model, optimizer)
    optimizer = OSS(params=model.parameters(), optim=optimizer)
    return model, optimizer

def process_long_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Process long text using Langchain's text splitter for improved NLP capabilities.

    Args:
        text (str): Long input text
        chunk_size (int): Maximum size of each text chunk (default: 1000)
        chunk_overlap (int): Number of characters to overlap between chunks (default: 200)

    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)
