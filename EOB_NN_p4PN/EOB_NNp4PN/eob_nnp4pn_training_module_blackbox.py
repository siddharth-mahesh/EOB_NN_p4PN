"""Black-box dissipative Hamiltonian neural network for RHS regression baselines."""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from EOB_NN_p4PN.mlp import MLP


class BlackBoxDHNN(eqx.Module):
    """Generic DHNN baseline.

    State convention:
    - Input sample: `[nu, r, phi, p_rstar, p_phi]`
    - Output RHS: `[dr/dt, dphi/dt, dp_rstar/dt, dp_phi/dt]`

    Model structure:
    - `H_theta(nu, y)` (scalar network) defines conservative flow via symplectic gradient.
    - `D_theta(nu, y)` (2-vector network) provides dissipative forcing on momentum channels.
    """

    h_net: MLP
    d_net: MLP
    h_scale: float
    d_scale: float

    def __init__(
        self,
        key: jax.Array = jax.random.PRNGKey(0),
        hidden_dim: int = 64,
        h_scale: float = 1.0,
        d_scale: float = 0.5,
    ):
        h_key, d_key = jax.random.split(key, 2)
        # Inputs are [nu, r, phi, p_rstar, p_phi].
        self.h_net = MLP(
            key=h_key,
            input_dim=5,
            output_dim=1,
            hidden_dim=hidden_dim,
            standalone=False,
        )
        self.d_net = MLP(
            key=d_key,
            input_dim=5,
            output_dim=2,
            hidden_dim=hidden_dim,
            standalone=False,
        )
        self.h_scale = h_scale
        self.d_scale = d_scale

    def _hamiltonian(self, nu: jax.Array, y: jax.Array) -> jax.Array:
        inp = jnp.concatenate([jnp.array([nu], dtype=y.dtype), y], axis=0)
        return self.h_scale * self.h_net(inp)[0]

    def _dissipation(self, nu: jax.Array, y: jax.Array) -> jax.Array:
        inp = jnp.concatenate([jnp.array([nu], dtype=y.dtype), y], axis=0)
        # Bound dissipation output for numerical robustness.
        return self.d_scale * jnp.tanh(self.d_net(inp))

    def _single_rhs(self, x_single: jax.Array) -> jax.Array:
        nu = x_single[0]
        y = x_single[1:]

        grad_h = jax.grad(lambda y_local: self._hamiltonian(nu, y_local))(y)
        # Canonical symplectic action J @ grad(H), for y=[q1, q2, p1, p2].
        cons = jnp.array([grad_h[2], grad_h[3], -grad_h[0], -grad_h[1]], dtype=y.dtype)

        diss_p = self._dissipation(nu, y)
        diss = jnp.array([0.0, 0.0, diss_p[0], diss_p[1]], dtype=y.dtype)
        return cons + diss

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self._single_rhs, in_axes=0)(x)


if __name__ == "__main__":
    model = BlackBoxDHNN(jax.random.PRNGKey(0), hidden_dim=64)
    x = jnp.load("seob_x_train_prelim.npy")[:8]
    y = model(x)
    print("BlackBoxDHNN output shape:", y.shape)
