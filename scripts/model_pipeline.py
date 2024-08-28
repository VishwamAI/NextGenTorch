import os
import torch
from torch.utils.data import Dataset, DataLoader
from nextgentorch import create_nextgentorch_model, setup_distributed_training
import yaml

# Load configurations
with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
    training_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/dataset_paths.yaml', 'r') as f:
    dataset_paths = yaml.safe_load(f)

class MathDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(dataset_name, split):
    data_path = os.path.join(dataset_paths['output'][dataset_name], f'{split}.pt')
    return MathDataset(data_path)

def create_data_loaders(dataset_name, batch_size):
    train_dataset = load_data(dataset_name, 'train')
    val_dataset = load_data(dataset_name, 'validation')
    test_dataset = load_data(dataset_name, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def main():
    # Create model
    model_size = training_config['model_sizes'][0]  # Using the first model size
    model = create_nextgentorch_model(model_size)

    # Load training dataset
    train_dataset = load_data('gsm8k', 'train')

    # Create validation and test data loaders
    batch_size = training_config['hyperparameters']['batch_size']
    _, val_loader, test_loader = create_data_loaders('gsm8k', batch_size)

    # Setup distributed training
    config_path = '/home/ubuntu/nextgentorch/config/deepspeed_config.json'  # Assuming this file exists
    model_engine, optimizer, train_dataloader = setup_distributed_training(model, train_dataset, config_path)

    # Basic training loop for testing
    num_epochs = training_config['hyperparameters']['num_epochs']
    for epoch in range(num_epochs):
        model_engine.train()
        for batch in train_dataloader:
            question_tokens, answer_tokens = batch
            outputs = model_engine(question_tokens)
            print(f"Outputs shape: {outputs.shape}")
            print(f"Answer tokens shape: {answer_tokens.shape}")
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), answer_tokens.view(-1))
            model_engine.backward(loss)
            model_engine.step()

        print(f"Epoch {epoch+1}/{num_epochs} completed")

    print("Training loop completed successfully")

if __name__ == "__main__":
    main()
