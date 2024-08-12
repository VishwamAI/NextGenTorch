# NextGenTorch

![Build Status](https://github.com/VishwamAI/NextGenTorch/workflows/Continuous%20Integration/badge.svg)
[![PyPI version](https://badge.fury.io/py/NextGenTorch.svg)](https://pypi.org/project/NextGenTorch/)

NextGenTorch is a powerful and flexible PyTorch-based library for next-generation language models.

## Installation

Install NextGenTorch and its dependencies using pip:

```bash
pip install NextGenTorch torch pytest fairscale langchain transformers
```

## Usage with Datasets

NextGenTorch supports seamless integration with datasets from various sources, including Hugging Face, Google, Meta, and Microsoft.

### Hugging Face Datasets

To use a dataset from Hugging Face:

```python
from datasets import load_dataset
from NextGenTorch import NextGenTorchModel

# Load a dataset
dataset = load_dataset("wikipedia", "20220301.en")

# Create a NextGenTorch model
model = NextGenTorchModel.from_pretrained("nextgentorch/base")

# Use the dataset with the model
for batch in dataset["train"]:
    outputs = model(batch["text"])
    # Process outputs as needed
```

### Google, Meta, and Microsoft Datasets

For datasets from Google, Meta, and Microsoft, please refer to their respective APIs and documentation. Once you have loaded the dataset, you can use it with NextGenTorch in a similar manner to the Hugging Face example above.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

NextGenTorch is released under the [MIT License](LICENSE).
