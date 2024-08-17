# NextGenTorch

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

## Generative AI Features

NextGenTorch includes powerful generative AI capabilities that allow users to create human-like text based on input prompts. Here are some key features:

1. **Text Generation**: Generate coherent and contextually relevant text using the `generate` method.
2. **Temperature Control**: Adjust the creativity and randomness of the generated text using the `temperature` parameter.
3. **Top-k Sampling**: Control the diversity of generated text with the `top_k` parameter.
4. **Beam Search**: Improve the quality of generated text using beam search with the `num_beams` parameter.
5. **Extended Context Length**: Handle longer contexts up to 8192 tokens, inspired by advanced models like Grok.

### Usage Example

```python
from NextGenTorch import ChatInterface

chat = ChatInterface(model_size="1b")
response = chat.chat("Tell me about artificial intelligence.", max_length=100, temperature=0.7, top_k=50)
print(response)
```

For more advanced usage and parameter tuning, please refer to the API documentation.
