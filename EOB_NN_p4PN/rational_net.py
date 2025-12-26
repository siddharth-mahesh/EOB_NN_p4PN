import jax
import jax.numpy as jnp

# set jax to 64 bit precision
jax.config.update("jax_enable_x64", True)
import equinox as eqx
from EOB_NN_p4PN.rational_activation import RationalActivation

class RationalNet(eqx.Module):
    extender: eqx.nn.Linear
    activation: RationalActivation
    scalarizer: eqx.nn.Linear
    """
    A Rational Neural Network model.
    This model serves as a building block 
    for the PN-mimicker neural networks built in this repo. 
    It is a Linear transformation that extends the input dimension
    followed by a rational activation function
    followed by another linear transformation that returns a scalar.
    """

    def __init__(self, key, input_dim, hidden_dim, degree_of_p, degree_of_q):
        """
        Initialize the RationalNet.

        Args:
        key (jax.random.PRNGKey): The random key for initialization.
        input_dim (int): The dimension of the input.
        hidden_dim (int): The dimension of the hidden layer.
        degree_of_p (int): The degree of the numerator polynomial.
        degree_of_q (int): The degree of the denominator polynomial.
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.extender = eqx.nn.Linear(input_dim, hidden_dim, key=key1)
        self.activation = RationalActivation(key=key2, features=hidden_dim, degree_of_p=degree_of_p, degree_of_q=degree_of_q)
        self.scalarizer = eqx.nn.Linear(hidden_dim, "scalar", key=key3)

    def __call__(self, x):
        """
        Compute the forward pass for a single input.
        """
        h = self.extender(x)
        h = self.activation(h)
        return self.scalarizer(h)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    key , key_x = jax.random.split(key, 2)
    rational_net = RationalNet(key, input_dim=2, hidden_dim=16, degree_of_p=2, degree_of_q=2)
    x = jax.random.normal(key_x, (2,))
    print(x.shape)
    print(rational_net(x).shape)
