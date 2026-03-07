"""Hybrid DHNN: EOB dynamics structure with fully learned A/D/Q/f potentials."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from EOB_NN_p4PN.EOB.eob_constants_3pn import set_eob_constants_3PN
from EOB_NN_p4PN.EOB.strain import strain
from EOB_NN_p4PN.mlp import MLP

from EOB_NN_p4PN.gamma import gamma as tgamma
jax.config.update("jax_enable_x64", True)
I = 1j


class ScalarPotentialHead(eqx.Module):
    """Two-input scalar head: inputs are [nu, x], output is one scalar."""

    net: MLP

    def __init__(self, key: jax.Array, input_dim, hidden_dim: int = 32, output_init_scale: float = 1e-3):
        self.net = MLP(
            key=key,
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            standalone=False,
        )
        output_init_scale = float(max(output_init_scale, 0.0))
        w_shape = self.net.lin_3.weight.shape
        b_shape = self.net.lin_3.bias.shape
        w_key, b_key = jax.random.split(key, 2)
        w_init = output_init_scale * jax.random.normal(w_key, w_shape, dtype=self.net.lin_3.weight.dtype)
        b_init = output_init_scale * jax.random.normal(b_key, b_shape, dtype=self.net.lin_3.bias.dtype)
        self.net = eqx.tree_at(lambda m: m.lin_3.weight, self.net, w_init)
        self.net = eqx.tree_at(lambda m: m.lin_3.bias, self.net, b_init)

    def __call__(self, neural_in: jax.Array) -> jax.Array:
        inp = jnp.array(neural_in, dtype=jnp.result_type(neural_in))
        return self.net(inp)[0]


class Hybrid_EOB_DHNN(eqx.Module):
    """EOB-form DHNN with unknown (learned) A, D, Q, f potentials.

    Structure is identical to the EOB dynamics flow:
    - conservative dynamics from Hamiltonian gradients
    - dissipative dynamics from flux
    but A/D/Q/f are black-box neural functions rather than PN-factorized models.
    """

    A_head: ScalarPotentialHead
    D_head: ScalarPotentialHead
    Q_head: ScalarPotentialHead
    f_head: ScalarPotentialHead
    _set_eob_constants_3PN: Callable
    srate: int

    A_floor: float
    D_floor: float
    Q_floor: float
    f_floor: float
    A_max: float
    D_max: float
    Q_max: float
    f_max: float

    def __init__(
        self,
        key: jax.Array = jax.random.PRNGKey(0),
        srate: int = 2000,
        hidden_dim_A: int = 32,
        hidden_dim_D: int = 32,
        hidden_dim_Q: int = 32,
        hidden_dim_f: int = 32,
        output_init_scale_A: float = 1e-3,
        output_init_scale_D: float = 1e-3,
        output_init_scale_Q: float = 1e-3,
        output_init_scale_f: float = 1e-3,
        A_floor: float = 1e-4,
        D_floor: float = 1e-4,
        Q_floor: float = 0.0,
        f_floor: float = 1e-4,
        A_max: float = 4.0,
        D_max: float = 4.0,
        Q_max: float = 8.0,
        f_max: float = 8.0,
    ):
        A_key, D_key, Q_key, f_key = jax.random.split(key, 4)
        self.A_head = ScalarPotentialHead(A_key, 2, hidden_dim_A, output_init_scale_A)
        self.D_head = ScalarPotentialHead(D_key, 2, hidden_dim_D, output_init_scale_D)
        self.Q_head = ScalarPotentialHead(Q_key, 3, hidden_dim_Q, output_init_scale_Q)
        self.f_head = ScalarPotentialHead(f_key, 2, hidden_dim_f, output_init_scale_f)
        self._set_eob_constants_3PN = set_eob_constants_3PN
        self.srate = srate

        self.A_floor = float(max(A_floor, 0.0))
        self.D_floor = float(max(D_floor, 0.0))
        self.Q_floor = float(max(Q_floor, 0.0))
        self.f_floor = float(max(f_floor, 0.0))
        self.A_max = float(max(A_max, self.A_floor + 1e-6))
        self.D_max = float(max(D_max, self.D_floor + 1e-6))
        self.Q_max = float(max(Q_max, self.Q_floor + 1e-6))
        self.f_max = float(max(f_max, self.f_floor + 1e-6))

    @staticmethod
    def _bounded_positive(raw: jax.Array, floor: float, vmax: float) -> jax.Array:
        pos = floor + jax.nn.softplus(raw)
        return jnp.clip(pos, floor, vmax)

    def _a_potential(self, r: jax.Array, nu: jax.Array) -> jax.Array:
        u = 1.0 / jnp.maximum(r, 1e-12)
        neural_in = jnp.array([nu, u], dtype=r.dtype)
        raw = self.A_head(neural_in)
        return self._bounded_positive(raw, self.A_floor, self.A_max)

    def _d_potential(self, r: jax.Array, nu: jax.Array) -> jax.Array:
        u = 1.0 / jnp.maximum(r, 1e-12)
        neural_in = jnp.array([nu, u], dtype=r.dtype)
        raw = self.D_head(neural_in)
        return self._bounded_positive(raw, self.D_floor, self.D_max)

    def _q_potential(self, prstar: jax.Array, r: jax.Array, nu: jax.Array) -> jax.Array:
        u = 1.0 / jnp.maximum(r, 1e-12)
        neural_in = jnp.array([nu, u, prstar], dtype=r.dtype)
        raw = self.Q_head(neural_in)
        return self._bounded_positive(raw, self.Q_floor, self.Q_max)

    def _f_potential(self, omega: jax.Array, nu: jax.Array) -> jax.Array:
        x = jnp.power(jnp.maximum(omega, 1e-12), 2.0 / 3.0)
        neural_in = jnp.array([nu, x], dtype=omega.dtype)
        raw = self.f_head(neural_in)
        return self._bounded_positive(raw, self.f_floor, self.f_max)

    def _strain(self, strain_qts: jax.Array, nu: jax.Array, constants) -> jax.Array:
        phi, hnu, omega = strain_qts
        omega_safe = jnp.maximum(omega, 1e-12)
        r0 = 2 / jnp.sqrt(jnp.e)
        tmp0 = jnp.pow(omega, 2.0 / 3.0)
        tmp2 = 4 * I * hnu * omega
        amp = self._f_potential(omega_safe, nu)
        h22 = (
            (4.0 / 5.0)
            * jnp.sqrt(5)
            * nu
            * jnp.sqrt(jnp.pi)
            * tmp0
            * (1 + (1.0 / 2.0) * (((hnu) * (hnu)) - 1) / nu)
            * amp
            * tgamma(3 - tmp2)
            * jnp.exp(-2 * I * phi)
            * jnp.exp(2 * hnu * omega * jnp.pi + tmp2 * jnp.log(4 * omega * r0))
        )
        return h22

    def _flux(self, strain_qts: jax.Array, nu: jax.Array, constants) -> jax.Array:
        omega = jnp.maximum(strain_qts[2], 1e-12)
        return -omega * jnp.abs(self._strain(strain_qts, nu, constants)) ** 2 / (2 * jnp.pi * nu)

    def _hamiltonian(self, y: jax.Array, nu: jax.Array):
        r, _, p_rstar, p_phi = y
        u = 1.0 / jnp.maximum(r, 1e-12)
        a = self._a_potential(r, nu)
        d = self._d_potential(r, nu)
        q = self._q_potential(p_rstar, r, nu)
        d_safe = jnp.maximum(d, 1e-12)
        xi = a / jnp.sqrt(d_safe)
        p_r = p_rstar / jnp.maximum(xi, 1e-12)

        inner_root = a * (
            (p_phi * p_phi) * (u * u)
            + (p_r * p_r) * (a / d_safe + (p_r * p_r) * (u * u) * q)
            + 1.0
        )
        inner_root_safe = jnp.maximum(inner_root, 1e-12)
        h_eff = jnp.sqrt(inner_root_safe)
        outer_root = 2.0 * nu * (h_eff - 1.0) + 1.0
        outer_root_safe = jnp.maximum(outer_root, 1e-12)
        h_real = jnp.sqrt(outer_root_safe) / nu
        return jnp.array([h_real, xi], dtype=y.dtype)

    def _single_rhs(self, x_single: jax.Array) -> jax.Array:
        nu = x_single[0]
        y = x_single[1:]

        num_coords = 2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords, num_coords)), jnp.eye(num_coords)],
                [-jnp.eye(num_coords), jnp.zeros((num_coords, num_coords))],
            ]
        )
        d_h_real = jax.jacfwd(self._hamiltonian, argnums=0)(y, nu)[0]
        h, xi = self._hamiltonian(y, nu)
        omega = d_h_real[3]
        constants = self._set_eob_constants_3PN(nu)
        strain_qts = jnp.array([y[1], h * nu, omega], dtype=y.dtype)
        flux = self._flux(strain_qts, nu, constants)

        ydot_cons_notort = symplectic_map @ d_h_real
        p_phi_safe = jnp.where(
            jnp.abs(y[3]) < 1e-12,
            jnp.where(y[3] >= 0.0, 1e-12, -1e-12),
            y[3],
        )
        ydot_flux = jnp.array([0.0, 0.0, flux * y[2] / p_phi_safe, flux], dtype=y.dtype)
        return jnp.array(
            [
                xi * ydot_cons_notort[0] + ydot_flux[0],
                ydot_cons_notort[1] + ydot_flux[1],
                xi * ydot_cons_notort[2] + ydot_flux[2],
                ydot_cons_notort[3] + ydot_flux[3],
            ],
            dtype=y.dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self._single_rhs, in_axes=0)(x)

