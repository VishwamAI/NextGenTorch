import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.oss import OSS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer  # Added for alternative tokenization

def preprocess_text(text: str) -> torch.Tensor:
    """
    Preprocess and tokenize input text using a basic tokenization method.

    Args:
        text (str): Input text to be preprocessed and tokenized.

    Returns:
        torch.Tensor: Tokenized and preprocessed text as a tensor.
    """
    # Use a simple whitespace tokenization
    tokens = text.split()
    # Convert tokens to integers (you may want to implement a more sophisticated method)
    token_ids = [hash(token) % 10000 for token in tokens]  # Using hash for simplicity
    return torch.tensor(token_ids)

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
    Set up Fairscale for distributed training using Fully Sharded Data Parallel (FSDP).

    Args:
        model: The PyTorch model
        optimizer: The optimizer

    Returns:
        tuple: (FSDP wrapped model, optimizer)
    """
    import torch.distributed as dist
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
        except RuntimeError as e:
            print(f"Warning: Failed to initialize process group: {e}")
            print("Continuing without distributed setup.")
            return model, optimizer

    if dist.is_initialized():
        try:
            model = FSDP(model)
        except Exception as e:
            print(f"Warning: Failed to wrap model with FSDP: {e}")
            print("Continuing with the original model.")

    # Note: FSDP doesn't require OSS optimizer, so we return the original optimizer
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
