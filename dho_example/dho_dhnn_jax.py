# dho_dhnn.py
"""
Dissipative Hamiltonian Neural Network (D-HNN) model for damped harmonic oscillator.
"""
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

    def __init__(self, key, input_dim, output_dim, hidden_dim,standalone= False):
        self.standalone = standalone
        key1, key2, key3 = jax.random.split(key, 3)
        self.lin_1 = eqx.nn.Linear(input_dim, hidden_dim, key=key1)
        self.lin_2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=key2)
        self.lin_3 = eqx.nn.Linear(hidden_dim, output_dim, key=key3)
    
    def _single_forward(self, x):
        h = jax.nn.tanh(self.lin_1(x))
        h = h + jax.nn.tanh(self.lin_2(h))
        return self.lin_3(h)

    def __call__(self, x):
        if not self.standalone:
            return self._single_forward(x)
        return jax.vmap(self._single_forward, in_axes=0)(x)

class DHNN_Model(eqx.Module):
    hnn: MLP
    dnn: MLP
    """
    Dissipative Hamiltonian Neural Network (D-HNN) model.
    """
    def __init__(self, key, input_dim=3, hidden_dim=256):
        """
        Initialize the D-HNN model.

        Args:
            input_dim (int): The dimension of the input.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int): The number of hidden layers.
        """
        # Two separate networks for H and D
        self.hnn = MLP(key, input_dim, 1, hidden_dim)
        self.dnn = MLP(key, input_dim, 1, hidden_dim)
    
    def _get_h(self, z):
        return self.hnn(z).squeeze()   

    def _get_d(self, z):
        return self.dnn(z).squeeze()   

    def _single_forward(self, z):
        input_dim = z.shape[0]
        num_canonical_variables = input_dim - 1
        half_size = num_canonical_variables//2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((half_size,half_size)), jnp.eye(half_size)],
                [-jnp.eye(half_size), jnp.zeros((half_size,half_size))]
            ]
        )
        dH = jax.grad(self._get_h, argnums=0)(z)
        dD = jax.grad(self._get_d, argnums=0)(z)
        dz = symplectic_map @ (dH[:num_canonical_variables]) + dD[:num_canonical_variables]
        return dz
    
    def _forward(self, z):
        return jax.vmap(self._single_forward, in_axes=0)(z)
    
    def __call__(self, z, rhs=True):
        """
        Defines the forward pass to compute the time derivatives (dq/dt, dp/dt).
        The forward pass is given by the dissipative Hamiltonian equations
        dq/dt = dH/dp + dD/dq
        dp/dt = -dH/dq + dD/dp

        Args:
            z (jax.numpy.ndarray): The state vector [q, p].
            rhs (bool): Whether to return the time derivatives [dq_dt, dp_dt] (default) or [H, D].

        Returns:
            jax.numpy.ndarray: The time derivatives [dq_dt, dp_dt] or [H, D] if rhs is False.
        """
        if rhs:
            return self._forward(z)
        else:
            return jnp.stack([jax.vmap(self._get_h, in_axes=0)(z), jax.vmap(self._get_d, in_axes=0)(z)], axis=-1)

if __name__=="__main__":
    SEED = 5678
    key , data_key = jax.random.split(jax.random.PRNGKey(SEED),2)
    dhnn = DHNN_Model(key=key)
    mlp = MLP(key=key, input_dim=3, output_dim=2, hidden_dim=256, standalone=True)
    z0 = jax.random.uniform(data_key, (100, 3), minval=-1, maxval=1)
    rhs = dhnn(z0)
    rhs_mlp = mlp(z0)
    print(rhs.shape)
    print(rhs_mlp.shape)

    