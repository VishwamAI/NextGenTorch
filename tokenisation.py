# Import necessary libraries
from neuroflex import *
import jax
import flax.linen as nn
import jax.numpy as jnp

# Define a simple tokenizer using Flax
class SimpleTokenizer(nn.Module):
    vocab_size: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=128)

    def __call__(self, x):
        return self.embedding(x)

# Example usage
def main():
    # Initialize the tokenizer
    tokenizer = SimpleTokenizer(vocab_size=10000)

    # Create random input data
    x = jax.random.randint(jax.random.PRNGKey(0), (10,), 0, 10000)

    # Apply the tokenizer
    y = tokenizer(x)
    print(y)

if __name__ == "__main__":
    main()
