import torch
import torch.nn as nn
import deepspeed
import yaml
from typing import Dict, Type

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

class NextGenTorchConfig:
    def __init__(self, layers, hidden_size, ff_size, attention_heads, vocab_size):
        self.layers = layers
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.attention_heads = attention_heads
        self.vocab_size = vocab_size

class NextGenTorchAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.attention_heads
        self.attention_head_size = config.hidden_size // config.attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value)

        return context_layer

class NextGenTorchLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = NextGenTorchAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.ff_size)
        self.output = nn.Linear(config.ff_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        hidden_states = self.layernorm1(hidden_states + attention_output)
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(hidden_states + layer_output)
        return layer_output

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NextGenTorchLayer(config) for _ in range(config.layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

MODEL_REGISTRY = {
    'gpt': GPTModel,
}

def create_nextgentorch_model():
    with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)
    with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)

    model_type = training_config['model']['type'].lower()
    model_size = training_config['model']['size']

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type. Choose from {', '.join(MODEL_REGISTRY.keys())}")

    if model_size not in model_config[model_type]:
        raise ValueError(f"Unsupported model size for {model_type}. Choose from {', '.join(model_config[model_type].keys())}")

    config = NextGenTorchConfig(**model_config[model_type][model_size])
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config)
    return model

def setup_distributed_training(model, train_dataset, config_path):
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=config_path
    )
    return model_engine, optimizer, train_dataloader
