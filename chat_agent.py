import os
from datasets import load_dataset
from ollama import Client

# API key
HF_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

class ChatAgent:
    def __init__(self, model_name="moondream"):
        self.client = Client()
        self.model_name = model_name

    def load_dataset(self, dataset_name, config_name):
        return load_dataset(dataset_name, config_name, use_auth_token=HF_API_KEY, trust_remote_code=True)

    def generate_response(self, prompt, max_length=100):
        response = self.client.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']

def main():
    agent = ChatAgent()
    dataset = agent.load_dataset("wikipedia", "20220301.en")

    # Example usage
    sample = dataset["train"][0]
    prompt = sample["text"][:100]  # Use first 100 characters as prompt

    # Generate response using Ollama
    response = agent.generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
