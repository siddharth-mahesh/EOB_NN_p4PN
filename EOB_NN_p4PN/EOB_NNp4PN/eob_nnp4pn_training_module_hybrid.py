"""
Hybrid Dissipative Hamiltonian Neural Network (DHNN).

This module implements a hybrid EOB dynamics structure where the overall physics flow
(conservative and dissipative equations of motion) is preserved, but the core 
potentials (A, D, Q, f) are fully learned as rational neural networks instead of 
PN-factorized baseline models.

Each potential head is a RationalNet: Linear → P(h)/Q(h) rational activation → Linear.
This produces a rational function of the inputs — structurally a Padé approximant —
matching the Padé-resummed form used by SEOBNRv5 for A and D.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from EOB_NN_p4PN.EOB.eob_constants_3pn import set_eob_constants_3PN
from EOB_NN_p4PN.EOB.strain import strain
from EOB_NN_p4PN.rational_net import RationalNet

from EOB_NN_p4PN.gamma import gamma as tgamma
jax.config.update("jax_enable_x64", True)
I = 1j


class Hybrid_EOB_DHNN(eqx.Module):
    """EOB-form DHNN with unknown (learned) A, D, Q, f potentials.

    Structure is identical to the EOB dynamics flow:
    - conservative dynamics from Hamiltonian gradients
    - dissipative dynamics from flux
    but A/D/Q/f are rational neural networks (Padé approximants) rather than
    PN-factorized models.

    Each head implements: Linear(input → hidden) → P(h)/Q(h) → Linear(hidden → 1)
    which produces a rational function of the inputs — matching SEOBNRv5's
    Padé-resummed A and D potentials by construction.
    """

    A_head: RationalNet
    D_head: RationalNet
    Q_head: RationalNet
    f_head: RationalNet
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
        degree_of_p: int = 4,
        degree_of_q: int = 5,
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

        Initializes the Hybrid DHNN with rational neural network heads.

        Each head is a RationalNet: Linear → P(h)/Q(h) → Linear, producing a rational
        function of its inputs (Padé approximant). degree_of_p / degree_of_q control
        the numerator / denominator polynomial degree of the rational activation.

        Input features (with ln included for logarithmic PN terms):
            A, D: [nu, u, ln(u)]            (input_dim=3)
            Q:    [nu, u, p_r*, ln(u)]      (input_dim=4)
            f:    [nu, v, ln(v)] where v=omega^(1/3) (input_dim=3)
        """
        A_key, D_key, Q_key, f_key = jax.random.split(key, 4)
        self.A_head = RationalNet(A_key, input_dim=3, hidden_dim=hidden_dim_A, degree_of_p=degree_of_p, degree_of_q=degree_of_q)
        self.D_head = RationalNet(D_key, input_dim=3, hidden_dim=hidden_dim_D, degree_of_p=degree_of_p, degree_of_q=degree_of_q)
        self.Q_head = RationalNet(Q_key, input_dim=4, hidden_dim=hidden_dim_Q, degree_of_p=degree_of_p, degree_of_q=degree_of_q)
        self.f_head = RationalNet(f_key, input_dim=3, hidden_dim=hidden_dim_f, degree_of_p=degree_of_p, degree_of_q=degree_of_q)
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

