import jax
import jax.numpy as jnp

# set jax to 64 bit precision
jax.config.update("jax_enable_x64", True)
import equinox as eqx


class MLP(eqx.Module):
    lin_1: eqx.nn.Linear
    lin_2: eqx.nn.Linear
    lin_3: eqx.nn.Linear
    standalone: bool
    """
    A Multi Layer Perceptron (MLP) model.
    This model serves as a building block 
    for the neural networks built in this repo. 
    It is a simple MLP with two hidden layers.
    """

    def __init__(self, key, input_dim, output_dim, hidden_dim, standalone=False):
        """
        Initialize the MLP.

        Args:
        key (jax.random.PRNGKey): The random key for initialization.
        input_dim (int): The dimension of the input.
        output_dim (int): The dimension of the output.
        hidden_dim (int): The dimension of the hidden layers.
        standalone (bool): Whether to use the MLP as a standalone model.
        """
        self.standalone = standalone
        key1, key2, key3 = jax.random.split(key, 3)
        self.lin_1 = eqx.nn.Linear(input_dim, hidden_dim, key=key1)
        self.lin_2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=key2)
        self.lin_3 = eqx.nn.Linear(hidden_dim, output_dim, key=key3)

    def _single_forward(self, x):
        """
        Compute the forward pass for a single input.
        """
        h = jax.nn.tanh(self.lin_1(x))
        h = jax.nn.tanh(self.lin_2(h))
        return self.lin_3(h)

    def __call__(self, x):
        """
        Compute the forward pass for a batch of inputs.
        """
        if not self.standalone:
            return self._single_forward(x)
        return jax.vmap(self._single_forward, in_axes=0)(x)