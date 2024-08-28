import os
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import GPT2Tokenizer
from nextgentorch import create_nextgentorch_model
import yaml

# Load configurations
with open('/home/ubuntu/nextgentorch/config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('/home/ubuntu/nextgentorch/config/training_config.yaml', 'r') as f:
    training_config = yaml.safe_load(f)

# Custom environment for RL training with LLM
class LLMEnvironment(gym.Env):
    def __init__(self, model, tokenizer):
        super(LLMEnvironment, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = 100
        self.current_step = 0

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(model.config.vocab_size)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(model.config.hidden_size,), dtype=float)

    def reset(self):
        self.current_step = 0
        # Start with a random hidden state
        obs = torch.randn(1, self.model.config.hidden_size)
        return obs.numpy().flatten()

    def step(self, action):
        self.current_step += 1

        # Ensure action is within the model's vocabulary size
        action = action % self.model.config.vocab_size

        # Convert action to token and get model output
        token = torch.tensor([[action]], dtype=torch.long)
        with torch.no_grad():
            output = self.model(token)

        # Use the last hidden state as the next observation
        obs = output[0][-1, :self.model.config.hidden_size]

        # Ensure obs has the correct shape
        obs = obs.reshape(self.model.config.hidden_size)

        # Simple reward function based on output probability
        reward = torch.softmax(output[0][-1], dim=-1).max().item()

        done = self.current_step >= self.max_steps

        return obs.numpy(), reward, done, {}

def setup_rl_training():
    # Create LLM model
    model_size = training_config['model_sizes'][0]
    llm_model = create_nextgentorch_model(model_size)

    # Load tokenizer (assuming GPT-2 tokenizer for this example)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create the custom environment
    env = LLMEnvironment(llm_model, tokenizer)

    # Wrap the environment in a DummyVecEnv for Stable-Baselines3 compatibility
    vec_env = DummyVecEnv([lambda: env])

    # Initialize the PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=1)

    return model, vec_env

def train_rl_model(model, env, total_timesteps=10000):
    model.learn(total_timesteps=total_timesteps)
    return model

def main():
    rl_model, env = setup_rl_training()
    trained_model = train_rl_model(rl_model, env)

    # Save the trained RL model
    trained_model.save("rl_llm_model")

    print("RL training completed and model saved.")

if __name__ == "__main__":
    main()
