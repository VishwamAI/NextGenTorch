# Import necessary libraries
from neuroflex import *
import jax
import flax.linen as nn
import jax.numpy as jnp

# Define a simple model using Flax
class SimpleModel(nn.Module):
    features: int

    def setup(self):
        self.dense = nn.Dense(features=self.features)

    def __call__(self, x):
        return self.dense(x)

# Example usage
def main():
    # Initialize the model
    model = SimpleModel(features=128)

    # Create random input data
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 10))

    # Apply the model
    y = model(x)
    print(y)

if __name__ == "__main__":
    main()
