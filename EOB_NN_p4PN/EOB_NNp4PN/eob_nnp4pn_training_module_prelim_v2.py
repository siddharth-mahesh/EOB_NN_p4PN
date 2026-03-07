"""
v2 preliminary Neural-EOB module.

This version replaces free-form per-potential rational networks with
coefficient heads over a PN+log basis:

    sum_k c_k(nu) * x^k + log(x) * sum_k d_k(nu) * x^k

where x is a potential-specific PN variable (u for A/D/Q, v_Omega^2 for f/delta).
The final layer of each coefficient head is initialized near zero so the model starts
close to the 3PN baseline while still allowing gradient flow into deeper layers.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
jax.config.update("jax_enable_x64", True)

from EOB_NN_p4PN.EOB.pade_1_3_a import pade_1_3
from EOB_NN_p4PN.EOB.pade_0_3_d import pade_0_3
from EOB_NN_p4PN.EOB.eob_constants_3pn import set_eob_constants_3PN
from EOB_NN_p4PN.EOB.strain import strain
from EOB_NN_p4PN.mlp import MLP


class PNLogCoefficientHead(eqx.Module):
    """Predicts PN basis coefficients with one log-power contribution."""

    net: MLP
    max_power: int

    def __init__(self, key, max_power: int = 3, hidden_dim: int = 16, output_init_scale: float = 1e-3):
        self.max_power = max_power
        output_dim = 2 * (max_power + 1)
        self.net = MLP(
            key=key,
            input_dim=1,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            standalone=False,
        )
        # Start near baseline (small non-zero coefficients) so gradients can
        # flow into deeper layers from step 0.
        output_init_scale = float(max(output_init_scale, 0.0))
        w_shape = self.net.lin_3.weight.shape
        b_shape = self.net.lin_3.bias.shape
        w_key, b_key = jax.random.split(key, 2)
        w_init = output_init_scale * jax.random.normal(w_key, w_shape, dtype=self.net.lin_3.weight.dtype)
        b_init = output_init_scale * jax.random.normal(b_key, b_shape, dtype=self.net.lin_3.bias.dtype)
        self.net = eqx.tree_at(
            lambda m: m.lin_3.weight,
            self.net,
            w_init,
        )
        self.net = eqx.tree_at(
            lambda m: m.lin_3.bias,
            self.net,
            b_init,
        )

    def __call__(self, x: jax.Array, nu: jax.Array) -> jax.Array:
        x_safe = jnp.maximum(x, 1e-12)
        dtype = x_safe.dtype
        powers = jnp.power(x_safe, jnp.arange(self.max_power + 1, dtype=dtype))
        coeffs = self.net(jnp.array([nu], dtype=dtype))
        n = self.max_power + 1
        poly_coeffs = coeffs[:n]
        log_coeffs = coeffs[n:]
        logx = jnp.log(x_safe)
        return jnp.dot(poly_coeffs, powers) + logx * jnp.dot(log_coeffs, powers)


class Neural_EOB_V2(eqx.Module):
    """
    Non-spinning 3PN EOB dynamics with trainable p4PN-style residual corrections.

    v2 design:
    - physics baseline remains analytic 3PN
    - neural corrections are coefficientized PN+log basis functions
    - turning scales to zero recovers the exact 3PN model
    """

    conservative_order: int
    radiative_order: float
    srate: int

    A_head: Optional[PNLogCoefficientHead]
    D_head: Optional[PNLogCoefficientHead]
    Q_head: Optional[PNLogCoefficientHead]
    f_head: Optional[PNLogCoefficientHead]
    delta_head: Optional[PNLogCoefficientHead]

    _pade_a: Callable
    _pade_d: Callable
    _set_eob_constants_3PN: Callable

    A_scale: float
    D_scale: float
    Q_scale: float
    f_scale: float
    delta_scale: float
    A_corr_bound: float
    D_corr_bound: float
    Q_corr_bound: float
    f_corr_bound: float
    delta_corr_bound: float
    enable_A: bool
    enable_D: bool
    enable_Q: bool
    enable_f: bool
    enable_delta: bool

    def __init__(
        self,
        key: jax.random.PRNGKey = jax.random.PRNGKey(42),
        srate: int = 2000,
        basis_order_A: int = 4,
        basis_order_D: int = 4,
        basis_order_Q: int = 4,
        basis_order_f: int = 4,
        basis_order_delta: int = 4,
        hidden_dim_A: int = 16,
        hidden_dim_D: int = 16,
        hidden_dim_Q: int = 16,
        hidden_dim_f: int = 16,
        hidden_dim_delta: int = 16,
        output_init_scale_A: float = 1e-3,
        output_init_scale_D: float = 1e-3,
        output_init_scale_Q: float = 1e-3,
        output_init_scale_f: float = 1e-3,
        output_init_scale_delta: float = 1e-3,
        A_corr_bound: float = 0.5,
        D_corr_bound: float = 0.1,
        Q_corr_bound: float = 2.,
        f_corr_bound: float = .5,
        delta_corr_bound: float = .5,
        enable_A: bool = True,
        enable_D: bool = True,
        enable_Q: bool = False,
        enable_f: bool = True,
        enable_delta: bool = False,
    ):
        self.conservative_order = 3
        self.radiative_order = 3.5
        self.srate = srate

        A_key, D_key, Q_key, f_key, delta_key = jax.random.split(key, 5)
        self.A_head = (
            PNLogCoefficientHead(
                A_key,
                max_power=basis_order_A,
                hidden_dim=hidden_dim_A,
                output_init_scale=output_init_scale_A,
            )
            if enable_A
            else None
        )
        self.D_head = (
            PNLogCoefficientHead(
                D_key,
                max_power=basis_order_D,
                hidden_dim=hidden_dim_D,
                output_init_scale=output_init_scale_D,
            )
            if enable_D
            else None
        )
        self.Q_head = (
            PNLogCoefficientHead(
                Q_key,
                max_power=basis_order_Q,
                hidden_dim=hidden_dim_Q,
                output_init_scale=output_init_scale_Q,
            )
            if enable_Q
            else None
        )
        self.f_head = (
            PNLogCoefficientHead(
                f_key,
                max_power=basis_order_f,
                hidden_dim=hidden_dim_f,
                output_init_scale=output_init_scale_f,
            )
            if enable_f
            else None
        )
        self.delta_head = (
            PNLogCoefficientHead(
                delta_key,
                max_power=basis_order_delta,
                hidden_dim=hidden_dim_delta,
                output_init_scale=output_init_scale_delta,
            )
            if enable_delta
            else None
        )

        self._set_eob_constants_3PN = set_eob_constants_3PN
        self._pade_a = pade_1_3
        self._pade_d = pade_0_3

        self.A_scale = 16.0
        self.D_scale = 16.0
        self.Q_scale = 16.0
        self.f_scale = 50.0
        self.delta_scale = 1.0
        self.A_corr_bound = A_corr_bound
        self.D_corr_bound = D_corr_bound
        self.Q_corr_bound = Q_corr_bound
        self.f_corr_bound = f_corr_bound
        self.delta_corr_bound = delta_corr_bound
        self.enable_A = enable_A
        self.enable_D = enable_D
        self.enable_Q = enable_Q
        self.enable_f = enable_f
        self.enable_delta = enable_delta

    def _bounded_corr(self, value: jax.Array, bound: float) -> jax.Array:
        return bound * jnp.tanh(value / bound)

    def _eval_head(
        self,
        head: Optional[PNLogCoefficientHead],
        x: jax.Array,
        nu: jax.Array,
        bound: float,
        enabled: bool,
    ) -> jax.Array:
        if (head is None) or (not enabled):
            return jnp.array(0.0, dtype=x.dtype)
        return self._bounded_corr(head(x, nu), bound)

    def _strain(self, strain_qts, nu, constants):
        phi, hnu, Omega = strain_qts
        Omega_safe = jnp.maximum(Omega, 1e-12)
        # v_Omega^2 ~ u for circular orbits; keep positive for log-basis input.
        x_rad = jnp.power(Omega_safe, 2.0 / 3.0)
        omega_pn = jnp.power(Omega_safe, 7.0 / 2.0)
        f_corr = self._eval_head(self.f_head, x_rad, nu, self.f_corr_bound, self.enable_f)
        delta_corr = self._eval_head(self.delta_head, x_rad, nu, self.delta_corr_bound, self.enable_delta)
        f_nn = 1.0 + omega_pn * self.f_scale * f_corr
        delta_nn = jnp.exp(1j * omega_pn * self.delta_scale * delta_corr)
        strain_qts_safe = jnp.array([phi, hnu, Omega_safe], dtype=strain_qts.dtype)
        return strain(self, strain_qts_safe, nu, constants) * f_nn * delta_nn

    def _flux(self, strain_qts, nu, constants):
        Omega = strain_qts[2]
        Omega_safe = jnp.maximum(Omega, 1e-12)
        return -Omega_safe * jnp.abs(self._strain(strain_qts, nu, constants)) ** 2 / (2 * jnp.pi * nu)

    def _a_potential(self, r, nu, constants):
        u = 1.0 / r
        correction = self._eval_head(self.A_head, u, nu, self.A_corr_bound, self.enable_A)
        baseline = self._pade_a(u, constants["a_1"], constants["a_3"], constants["a_4"])
        return baseline * (1.0 + nu * jnp.power(u, 5) * self.A_scale * correction)

    def _d_potential(self, r, nu, constants):
        u = 1.0 / r
        correction = self._eval_head(self.D_head, u, nu, self.D_corr_bound, self.enable_D)
        baseline = self._pade_d(u, constants["d_2"], constants["d_3"])
        return baseline * (1.0 + nu * jnp.power(u, 4) * self.D_scale * correction)

    def _hamiltonian(self, y, nu, constants):
        r, _, p_rstar, p_phi = y
        u = 1.0 / r
        z_3 = constants["z_3"]
        a = self._a_potential(r, nu, constants)
        d = self._d_potential(r, nu, constants)
        d_safe = jnp.maximum(d, 1e-12)
        xi = a / jnp.sqrt(d_safe)
        p_r = p_rstar / xi
        q_corr = self._eval_head(self.Q_head, u, nu, self.Q_corr_bound, self.enable_Q)
        q_p4pn = 1.0 + jnp.power(u, 4) * self.Q_scale * q_corr
        inner_root = a * (
            (p_phi * p_phi) * (u * u)
            + (p_r * p_r) * (a / d_safe + (p_r * p_r) * ((u * u) * z_3) * q_p4pn)
            + 1.0
        )
        inner_root_safe = jnp.maximum(inner_root, 1e-12)
        outer_root = 2.0 * nu * (jnp.sqrt(inner_root_safe) - 1.0) + 1.0
        outer_root_safe = jnp.maximum(outer_root, 1e-12)
        h_real = jnp.sqrt(outer_root_safe) / nu
        return jnp.array([h_real, xi])

    def _eom(self, t, y, args):
        del t
        nu, constants = args
        num_coords = 2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords, num_coords)), jnp.eye(num_coords)],
                [-jnp.eye(num_coords), jnp.zeros((num_coords, num_coords))],
            ]
        )
        d_h_real = jax.jacfwd(self._hamiltonian, argnums=0)(y, nu, constants)[0]
        h, xi = self._hamiltonian(y, nu, constants)
        omega = d_h_real[3]
        strain_qts = jnp.array([y[1], h * nu, omega])
        flux = self._flux(strain_qts, nu, constants)
        ydot_cons_notort = symplectic_map @ d_h_real
        p_phi_safe = jnp.where(
            jnp.abs(y[3]) < 1e-12,
            jnp.where(y[3] >= 0.0, 1e-12, -1e-12),
            y[3],
        )
        ydot_flux = jnp.array([0.0, 0.0, flux * y[2] / p_phi_safe, flux])
        return jnp.array(
            [
                xi * ydot_cons_notort[0] + ydot_flux[0],
                ydot_cons_notort[1] + ydot_flux[1],
                xi * ydot_cons_notort[2] + ydot_flux[2],
                ydot_cons_notort[3] + ydot_flux[3],
            ]
        )

    def _single_pass_training(self, x):
        nu = x[0]
        prims = x[1:]
        constants = self._set_eob_constants_3PN(nu)
        return self._eom(0.0, prims, (nu, constants))

    def photon_effective_potential(self, r_grid, nu):
        constants = self._set_eob_constants_3PN(nu)
        a = jax.vmap(self._a_potential, in_axes=(0, None, None))(r_grid, nu, constants)
        return a / r_grid**2

    def particle_effective_potential(self, r_grid, j_grid, nu):
        constants = self._set_eob_constants_3PN(nu)
        a = jax.vmap(self._a_potential, in_axes=(0, None, None))(r_grid, nu, constants)
        return a * (1.0 + j_grid / r_grid**2)

    def __call__(self, x):
        return jax.vmap(self._single_pass_training, in_axes=(0))(x)


# Convenience alias for trainer imports.
Neural_EOB = Neural_EOB_V2

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model = Neural_EOB(key)
    @eqx.filter_jit
    def test_grad(model, x, y):
        y_pred = model(x)
        return jnp.mean(jnp.abs((y-y_pred)**2))

    print("Testing grad")
    x = jnp.load("seob_x_train_prelim.npy")
    y = jnp.load("seob_y_train_prelim.npy")
    try:
        val, grad = eqx.filter_value_and_grad(test_grad)(model, x, y)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad)))
        print("Value:", val)
        print("Grad Norm:", grad_norm)
    except Exception as e:
        print("Error:", e)
