"""
Hybrid Dissipative Hamiltonian Neural Network (DHNN).

This module implements a hybrid EOB dynamics structure where the overall physics flow
(conservative and dissipative equations of motion) is preserved, but the core 
potentials (A, D, Q, f) are fully learned using neural networks instead of 
PN-factorized baseline models.
"""

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
    """two_input_scalar_head
    
    Two-input scalar head: inputs are [nu, x], output is one scalar.
    """

    net: MLP

    def __init__(self, key: jax.Array, input_dim: int, hidden_dim: int = 32, depth: int = 2, output_init_scale: float = 1e-3):
        """initialize_scalar_head
        
        Initializes the scalar potential head with an parameterized MLP layer.
        """
        self.net = MLP(
            key=key,
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=depth,
            standalone=False,
        )
        output_init_scale = float(max(output_init_scale, 0.0))
        w_shape = self.net.layers[-1].weight.shape
        b_shape = self.net.layers[-1].bias.shape
        w_key, b_key = jax.random.split(key, 2)
        w_init = output_init_scale * jax.random.normal(w_key, w_shape, dtype=self.net.layers[-1].weight.dtype)
        b_init = output_init_scale * jax.random.normal(b_key, b_shape, dtype=self.net.layers[-1].bias.dtype)
        
        # eqx.tree_at replaces the leaves. For tuple, we can use a small function.
        def replace_last_layer_weight(m):
            return m.layers[-1].weight
        def replace_last_layer_bias(m):
            return m.layers[-1].bias

        self.net = eqx.tree_at(replace_last_layer_weight, self.net, w_init)
        self.net = eqx.tree_at(replace_last_layer_bias, self.net, b_init)

    def __call__(self, neural_in: jax.Array) -> jax.Array:
        """call
        
        Compute the forward evaluation.
        """
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
    _set_eob_constants_3PN: Callable = eqx.field(static=True)
    srate: int

    A_floor: float = eqx.field(static=True)
    D_floor: float = eqx.field(static=True)
    Q_floor: float = eqx.field(static=True)
    f_floor: float = eqx.field(static=True)
    A_max: float = eqx.field(static=True)
    D_max: float = eqx.field(static=True)
    Q_max: float = eqx.field(static=True)
    f_max: float = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array = jax.random.PRNGKey(0),
        srate: int = 2000,
        hidden_dim_A: int = 32,
        hidden_dim_D: int = 32,
        hidden_dim_Q: int = 32,
        hidden_dim_f: int = 32,
        depth_A: int = 2,
        depth_D: int = 2,
        depth_Q: int = 2,
        depth_f: int = 2,
        output_init_scale_A: float = 1e-3,
        output_init_scale_D: float = 1e-3,
        output_init_scale_Q: float = 1e-3,
        output_init_scale_f: float = 1e-3,
        A_floor: float = 1e-4,
        D_floor: float = 1e-4,
        Q_floor: float = 0.0,
        f_floor: float = 1e-4,
        A_max: float = 20.0,
        D_max: float = 20.0,
        Q_max: float = 20.0,
        f_max: float = 20.0,
    ):
        """initialize_hybrid_dhnn
        
        Initializes the Hybrid DHNN with tunable depth per potential and relaxed output ceilings.
        """
        A_key, D_key, Q_key, f_key = jax.random.split(key, 4)
        self.A_head = ScalarPotentialHead(A_key, 3, hidden_dim_A, depth_A, output_init_scale_A)
        self.D_head = ScalarPotentialHead(D_key, 3, hidden_dim_D, depth_D, output_init_scale_D)
        self.Q_head = ScalarPotentialHead(Q_key, 4, hidden_dim_Q, depth_Q, output_init_scale_Q)
        self.f_head = ScalarPotentialHead(f_key, 3, hidden_dim_f, depth_f, output_init_scale_f)
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
        """bounded_positive
        
        Apply a bounding function that clips raw output via sigmoid and an upper bound limit.
        """
        return floor + (vmax - floor) * jax.nn.sigmoid(raw)

    def _a_potential(self, r: jax.Array, nu: jax.Array) -> jax.Array:
        """a_potential
        
        Compute the conservative A potential.
        """
        u = 1.0 / jnp.maximum(r, 1e-12)
        # ln(u) is explicitly provided so the MLP need not reconstruct it from u.
        # The SEOBNRv5 A potential has ln(u) terms at 5PN order.
        neural_in = jnp.array([nu, u, jnp.log(u)], dtype=r.dtype)
        raw = self.A_head(neural_in)
        return self._bounded_positive(raw, self.A_floor, self.A_max)

    def _d_potential(self, r: jax.Array, nu: jax.Array) -> jax.Array:
        """d_potential
        
        Compute the conservative D potential.
        """
        u = 1.0 / jnp.maximum(r, 1e-12)
        # ln(u) is explicitly provided for the same reason as in _a_potential.
        neural_in = jnp.array([nu, u, jnp.log(u)], dtype=r.dtype)
        raw = self.D_head(neural_in)
        return self._bounded_positive(raw, self.D_floor, self.D_max)

    def _q_potential(self, prstar: jax.Array, r: jax.Array, nu: jax.Array) -> jax.Array:
        """q_potential
        
        Compute the strong-field modifying Q potential.
        """
        u = 1.0 / jnp.maximum(r, 1e-12)
        # ln(u) added for consistency with A and D; Q also acquires log corrections at high PN.
        neural_in = jnp.array([nu, u, prstar, jnp.log(u)], dtype=r.dtype)
        raw = self.Q_head(neural_in)
        return self._bounded_positive(raw, self.Q_floor, self.Q_max)

    def _f_potential(self, omega: jax.Array, nu: jax.Array) -> jax.Array:
        """f_potential
        
        Compute the f amplitude modifier for strain and flux.
        """
        x = jnp.power(jnp.maximum(omega, 1e-12), 1.0 / 3.0)
        neural_in = jnp.array([nu, x, jnp.log(x)], dtype=omega.dtype)
        raw = self.f_head(neural_in)
        return self._bounded_positive(raw, self.f_floor, self.f_max)

    def _strain(self, strain_qts: jax.Array, nu: jax.Array, constants) -> jax.Array:
        """calculate_strain
        
        Calculate the modified leading mode 22 strain emission from potentials.
        """
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
        """calculate_flux
        
        Calculate the energy flux from the leading mode strain.
        """
        omega = jnp.maximum(strain_qts[2], 1e-12)
        return -omega * jnp.abs(self._strain(strain_qts, nu, constants)) ** 2 / (2 * jnp.pi * nu)

    def _hamiltonian(self, y: jax.Array, nu: jax.Array):
        """calculate_hamiltonian
        
        Calculate the real EOB Hamiltonian and scale factor xi.
        """
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
        """calculate_single_rhs
        
        Calculate the right-hand side equations of motion for a single system state.
        """
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
        """call
        
        Calculate the batched right-hand side equations of motion.
        """
        return jax.vmap(self._single_rhs, in_axes=0)(x)

