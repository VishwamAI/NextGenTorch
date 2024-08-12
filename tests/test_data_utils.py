import pytest
import torch
import unittest
from unittest.mock import patch
from utils.data_utils import (
    preprocess_text,
    create_data_loader,
    pad_sequences,
    setup_fairscale,
    process_long_text
)

def test_preprocess_text():
    text = "Hello world! This is a test sentence."
    result = preprocess_text(text)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 1
    assert result.numel() > 6  # Number of tokens should be greater than number of words
    assert result.dtype == torch.long  # Ensure the output is of integer type

    # Test with empty string
    empty_result = preprocess_text("")
    assert empty_result.numel() == 0

    # Test with long text
    long_text = "This is a longer text with more words to test the tokenization process."
    long_result = preprocess_text(long_text)
    assert long_result.numel() > 12  # Number of tokens should be greater than number of words

    # Test token count consistency
    assert result.numel() == len(text.split())  # Simple word-based tokenization
    assert long_result.numel() == len(long_text.split())  # Simple word-based tokenization

def test_create_data_loader():
    dataset = torch.randn(100, 10)
    batch_size = 16

    # Test without distributed training
    loader = create_data_loader(dataset, batch_size)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert loader.batch_size == batch_size

    # Test with distributed training using mock objects
    with unittest.mock.patch('torch.distributed.is_initialized', return_value=True):
        with unittest.mock.patch('torch.utils.data.distributed.DistributedSampler') as mock_sampler:
            with unittest.mock.patch('torch.distributed.get_world_size', return_value=2):
                with unittest.mock.patch('torch.distributed.get_rank', return_value=0):
                    distributed_loader = create_data_loader(dataset, batch_size, distributed=True)
                    assert isinstance(distributed_loader, torch.utils.data.DataLoader)
                    mock_sampler.assert_called_once_with(dataset)

def test_pad_sequences():
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    padded = pad_sequences(sequences)
    assert isinstance(padded, torch.Tensor)
    assert padded.shape == (3, 4)
    assert torch.all(padded[-1] == torch.tensor([6, 7, 8, 9]))

def test_setup_fairscale():
    import torch.distributed as dist
    from unittest.mock import patch, MagicMock
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

    # Create a mock environment for torch.distributed
    mock_env = {
        'is_initialized': False,
        'world_size': 2,
        'rank': 0,
    }

    def mock_is_initialized():
        return mock_env['is_initialized']

    def mock_init_process_group(backend):
        mock_env['is_initialized'] = True

    def mock_get_world_size():
        return mock_env['world_size']

    def mock_get_rank():
        return mock_env['rank']

    # Use patch to mock the torch.distributed functions
    with patch.multiple(dist,
                        is_initialized=mock_is_initialized,
                        init_process_group=mock_init_process_group,
                        get_world_size=mock_get_world_size,
                        get_rank=mock_get_rank):

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Test when process group is not initialized
        fsdp_model, returned_optimizer = setup_fairscale(model, optimizer)
        assert not isinstance(fsdp_model, FSDP)
        assert fsdp_model is model
        assert returned_optimizer is optimizer

        # Initialize the process group
        dist.init_process_group(backend='gloo')
        assert mock_env['is_initialized'] == True

        # Test successful FSDP wrapping
        with patch('fairscale.nn.data_parallel.FullyShardedDataParallel', return_value=MagicMock(spec=FSDP)) as mock_fsdp:
            fsdp_model, returned_optimizer = setup_fairscale(model, optimizer)
            assert isinstance(fsdp_model, FSDP)
            assert returned_optimizer is optimizer
            mock_fsdp.assert_called_once_with(model)

        # Test FSDP wrapping failure
        with patch('fairscale.nn.data_parallel.FullyShardedDataParallel', side_effect=Exception("FSDP wrapping failed")):
            fsdp_model, returned_optimizer = setup_fairscale(model, optimizer)
            assert not isinstance(fsdp_model, FSDP)
            assert fsdp_model is model
            assert returned_optimizer is optimizer

    # The mock environment is automatically cleaned up when exiting the context manager

def test_process_long_text():
    long_text = "This is a very long text. " * 100
    chunks = process_long_text(long_text, chunk_size=100, chunk_overlap=20)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
