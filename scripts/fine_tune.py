import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from nextgentorch import create_nextgentorch_model
import yaml
from model_pipeline import MathDataset, load_data

# Load configurations
with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
    training_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/dataset_paths.yaml', 'r') as f:
    dataset_paths = yaml.safe_load(f)

def create_data_loaders(dataset_name, batch_size):
    train_dataset = load_data(dataset_name, 'train')
    val_dataset = load_data(dataset_name, 'validation')
    test_dataset = load_data(dataset_name, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def fine_tune():
    # Create model
    model_size = training_config['model_sizes'][0]  # Using the first model size
    model = create_nextgentorch_model(model_size)

    # Create data loaders
    batch_size = training_config['hyperparameters']['batch_size']
    train_loader, val_loader, _ = create_data_loaders('gsm8k', batch_size)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=training_config['hyperparameters']['num_epochs'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=training_config['hyperparameters']['warmup_steps'],
        weight_decay=training_config['hyperparameters']['weight_decay'],
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    fine_tune()
