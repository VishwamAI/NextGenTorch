from nextgenjax.layers import DenseLayer, ConvolutionalLayer, TransformerLayer
import torch
import tensorflow as tf
import jax.numpy as jnp
from jax import device_put
import numpy as np

# Create a wrapper class for the NextGenJAX DenseLayer to be used in PyTorch and TensorFlow models.
class DenseLayerWrapper:
    def __init__(self, features, activation=None):
        # Initialize the NextGenJAX DenseLayer with the provided features and activation function.
        self.dense_layer = DenseLayer(features=features, activation=activation)

    def __call__(self, x):
        # Convert input from PyTorch/TensorFlow tensor to JAX tensor.
        x_jax = convert_to_jax(x)
        # Apply the NextGenJAX DenseLayer.
        y_jax = self.dense_layer(x_jax)
        # Convert output from JAX tensor back to PyTorch/TensorFlow tensor.
        y = convert_from_jax(y_jax)
        return y

class ConvolutionalLayerWrapper:
    def __init__(self, features, kernel_size, strides=None, padding='SAME', activation=None):
        # Initialize the NextGenJAX ConvolutionalLayer with the provided parameters.
        self.conv_layer = ConvolutionalLayer(features=features, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)

    def __call__(self, x):
        # Convert input from PyTorch/TensorFlow tensor to JAX tensor.
        x_jax = convert_to_jax(x)
        # Apply the NextGenJAX ConvolutionalLayer.
        y_jax = self.conv_layer(x_jax)
        # Convert output from JAX tensor back to PyTorch/TensorFlow tensor.
        y = convert_from_jax(y_jax)
        return y

class TransformerLayerWrapper:
    def __init__(self, d_model, num_heads, d_ff=None, dropout_rate=0.1, activation='relu'):
        # Initialize the NextGenJAX TransformerLayer with the provided parameters.
        self.transformer_layer = TransformerLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout_rate=dropout_rate, activation=activation)

    def __call__(self, x):
        # Convert input from PyTorch/TensorFlow tensor to JAX tensor.
        x_jax = convert_to_jax(x)
        # Apply the NextGenJAX TransformerLayer.
        y_jax = self.transformer_layer(x_jax)
        # Convert output from JAX tensor back to PyTorch/TensorFlow tensor.
        y = convert_from_jax(y_jax)
        return y

def convert_to_jax(tensor):
    # This function converts PyTorch or TensorFlow tensors to JAX tensors.
    if isinstance(tensor, torch.Tensor):
        # Convert PyTorch tensor to numpy array, then to JAX tensor.
        return device_put(tensor.numpy())
    elif isinstance(tensor, tf.Tensor):
        # Convert TensorFlow tensor to numpy array, then to JAX tensor.
        return device_put(tensor.numpy())
    else:
        raise TypeError("Input tensor must be a PyTorch or TensorFlow tensor.")

def convert_from_jax(jax_tensor):
    # This function converts JAX tensors to PyTorch tensors.
    # Note: The choice of PyTorch here is arbitrary; in a complete implementation,
    # we would include logic to convert to the appropriate framework's tensor type.
    return torch.from_numpy(np.array(jax_tensor))
