import jax
import jax.numpy as jnp

# set jax to 64 bit precision
jax.config.update("jax_enable_x64", True)
import equinox as eqx


class MLP(eqx.Module):
    """A Multi Layer Perceptron (MLP) model.
    This model serves as a building block 
    for the neural networks built in this repo. 
    It is a parameterized MLP.
    """
    layers: tuple
    standalone: bool

    def __init__(self, key, input_dim, output_dim, hidden_dim, depth=2, standalone=False):
        """initialize_the_mlp.

        Initialize the MLP.

        Args:
            key (jax.random.PRNGKey): The random key for initialization.
            input_dim (int): The dimension of the input.
            output_dim (int): The dimension of the output.
            hidden_dim (int): The dimension of the hidden layers.
            depth (int): Number of hidden layers. Defaults to 2.
            standalone (bool): Whether to use the MLP as a standalone model.
        """
        self.standalone = standalone
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(eqx.nn.Linear(input_dim, output_dim, key=keys[0]))
        else:
            layers.append(eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]))
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i + 1]))
            layers.append(eqx.nn.Linear(hidden_dim, output_dim, key=keys[-1]))
        
        self.layers = tuple(layers)

    def _single_forward(self, x):
        """single_forward
        
        Compute the forward pass for a single input.
        """
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = jax.nn.tanh(layer(h))
        return self.layers[-1](h)

    def __call__(self, x):
        """call
        
        Compute the forward pass for a batch of inputs natively.
        """
        if not self.standalone:
            return self._single_forward(x)
        return jax.vmap(self._single_forward, in_axes=0)(x)