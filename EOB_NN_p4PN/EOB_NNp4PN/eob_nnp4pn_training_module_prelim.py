"""
Preliminary Training Module for Neural EOB.

This module implements an earlier or alternative version of the `Neural_EOB` class 
for training with post-4PN terms. Uses scaled rational networks to fit the EOB potentials 
and provides RHS predictions for the Dissipative Hamiltonian system.
"""

I = 1j
import jax
import jax.numpy as jnp
from jax.numpy import log
from EOB_NN_p4PN.EOB.flux import flux
from EOB_NN_p4PN.EOB.pade_1_3_a import pade_1_3
from EOB_NN_p4PN.EOB.pade_0_3_d import pade_0_3
from EOB_NN_p4PN.EOB.eob_constants_3pn import set_eob_constants_3PN
from EOB_NN_p4PN.EOB.strain import strain
from EOB_NN_p4PN.EOB_3PN.eob3pn import EOB as EOB_3PN
# set jax to 64 bit precision
jax.config.update("jax_enable_x64", True)
import diffrax
import optimistix
import equinox as eqx

#from EOB_NN_p4PN.mlp import MLP
from EOB_NN_p4PN.rational_net import RationalNet
from typing import Callable


class Neural_EOB(eqx.Module):
    """
    This class implements the non-spinning 3PN Effective One Body model
    with 3.5PN circular radiation-reaction
    and neural post-4PN terms.
    """

    conservative_order: int
    radiative_order: float
    srate: int
    A_p4PN: RationalNet
    D_p4PN: RationalNet
    Q_p4PN: RationalNet
    f_p4PN: RationalNet
    delta_p4PN: RationalNet
    _flux: Callable
    _pade_a: Callable
    _pade_d: Callable
    _set_eob_constants_3PN: Callable
    _strain: Callable
    A_scale: float
    D_scale: float
    Q_scale: float
    f_scale: float
    delta_scale: float

    def __init__(
        self,
        key:jax.random.PRNGKey=jax.random.PRNGKey(42),
        srate:int=2000,
        hidden_dim_A:int=20,
        hidden_dim_D:int=20,
        hidden_dim_Q:int=20,
        hidden_dim_f:int=20,
        hidden_dim_delta:int=20,
    ):
        """
        Initialize the EOB class.
        Args:
            key (jax.random.PRNGKey): The random key for initialization.
            srate (int): The sampling rate for the strain.
            hidden_dim_A (int): The dimension of the hidden layer for A potential.
            hidden_dim_D (int): The dimension of the hidden layer for D potential.
            hidden_dim_Q (int): The dimension of the hidden layer for Q potential.
            hidden_dim_f (int): The dimension of the hidden layer for f potential.
            hidden_dim_delta (int): The dimension of the hidden layer for delta potential.
        """

        # model identifiers
        self.conservative_order = 3
        self.radiative_order = 3.5
        self.srate = srate
        A_key, D_key, Q_key, f_key, delta_key = jax.random.split(key, 5)
        self.A_p4PN = RationalNet(
            key=A_key,
            input_dim=2,
            hidden_dim=hidden_dim_A,
            degree_of_p=2,
            degree_of_q=3,
        )
        self.D_p4PN = RationalNet(
            key=D_key,
            input_dim=2,
            hidden_dim=hidden_dim_D,
            degree_of_p=2,
            degree_of_q=3,
        )
        self.Q_p4PN = RationalNet(
            key=Q_key,
            input_dim=3,
            hidden_dim=hidden_dim_Q,
            degree_of_p=2,
            degree_of_q=3,
        )
        self.f_p4PN = RationalNet(
            key=f_key,
            input_dim=2,
            hidden_dim=hidden_dim_f,
            degree_of_p=2,
            degree_of_q=3,
        )
        self.delta_p4PN = RationalNet(
            key=delta_key,
            input_dim=2,
            hidden_dim=hidden_dim_delta,
            degree_of_p=2,
            degree_of_q=3,
        )
        self._set_eob_constants_3PN = set_eob_constants_3PN
        self._pade_a = pade_1_3
        self._pade_d = pade_0_3
        self.A_scale = 16.
        self.D_scale = 16.
        self.Q_scale = 16.
        self.f_scale = 50.
        self.delta_scale = 1.

    def _strain(self, strain_qts, nu, constants):
        Omega = strain_qts[2] 
        f_nn = 1 + jnp.pow(Omega,7/2) * self.f_scale*self.f_p4PN(jnp.array([Omega, nu]))
        delta_nn = jnp.exp(1j*jnp.pow(Omega,7/2) * self.delta_scale*self.delta_p4PN(jnp.array([Omega, nu])))
        return strain(self,strain_qts, nu, constants) * f_nn * delta_nn

    def _flux(self, strain_qts, nu, constants):
        Omega = strain_qts[2]
        return -Omega*jnp.abs(self._strain(strain_qts, nu, constants))**2/(2*jnp.pi*nu)

    def _a_potential(self, r, nu, constants):
        """
        Compute the Hamiltonian A potential.

        Args:
            r (float): Radial position
            nu (float): Symmetric mass ratio
            constants (dict): Dictionary of constants.

        Returns:
            float: Hamiltonian A potential
        """
        u = 1 / r
        neural_in = jnp.array([u, nu])
        a = self._pade_a(u, constants["a_1"], constants["a_3"], constants["a_4"]) * (
            1 + nu*jnp.pow(u,5)*self.A_scale*self.A_p4PN(neural_in)
        )
        return a

    def _d_potential(self, r, nu, constants):
        """
        Compute the Hamiltonian D potential.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Hamiltonian D potential
        """
        u = 1 / r
        neural_in = jnp.array([u, nu])
        d_nn = 1 + nu * jnp.pow(u,4)*self.D_scale*self.D_p4PN(neural_in)
        d = self._pade_d(u, constants["d_2"], constants["d_3"])*d_nn 
        return d

    def _hamiltonian(self, y, nu, constants):
        """
        Compute the Hamiltonian.

        Args:
            y (jnp.ndarray): Canonical variables [r, phi, p_r, p_phi].
            nu (float): Symmetric mass ratio.
            constants (dict): Dictionary of constants.

        Returns:
            float: Hamiltonian evaluated at y.
        """
        r, phi, p_rstar, p_phi = y
        u = 1 / r
        z_3 = constants["z_3"]
        a = self._a_potential(r, nu, constants)
        d = self._d_potential(r, nu, constants)
        xi = a/jnp.sqrt(d)
        p_r = p_rstar/xi
        neural_q_in = jnp.array([nu, u, p_r])
        q_p4pn = 1 + jnp.pow(u,4)*self.Q_scale*self.Q_p4PN(neural_q_in)
        inner_root = (a* (
                ((p_phi) * (p_phi)) * ((u) * (u))
                + ((p_r) * (p_r))
                * (
                    a / d
                    + ((p_r) * (p_r)) * (((u) * (u)) * z_3)*q_p4pn
                )
                + 1
            )
        )
        outer_root = 2 * nu * (jnp.sqrt(inner_root) - 1) + 1
        h_real = jnp.sqrt(outer_root) / nu
        return jnp.array([h_real , xi])
    
    def _eom(self, t, y, args):
        """
        The equations of motion for the EOB model.

        Args:
            t (float): Time.
            y (jnp.ndarray): Canonical variables [r, phi, p_rstar, p_phi].
            args (tuple): Additional parameters (nu, constants).

        Returns:
            jnp.ndarray: Equations of motion.
        """
        nu, constants = args
        num_coords = 2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords, num_coords)), jnp.eye(num_coords)],
                [-jnp.eye(num_coords), jnp.zeros((num_coords, num_coords))],
            ]
        )
        d_h_real = jax.jacfwd(self._hamiltonian, argnums=0)(y, nu, constants)[0]
        h , xi = self._hamiltonian(y,nu,constants)
        omega = d_h_real[3]  # omega = d_h_real/d_p_phi
        strain_qts = jnp.array([y[1], h*nu, omega])
        flux = self._flux(strain_qts, nu, constants)
        ydot_cons_notort = symplectic_map @ d_h_real 
        ydot_flux = jnp.array([0.0, 0.0, flux*y[2]/y[3], flux])
        ydot = jnp.array([
            xi*ydot_cons_notort[0] + ydot_flux[0],
            ydot_cons_notort[1] + ydot_flux[1], 
            xi*ydot_cons_notort[2] + ydot_flux[2], 
            ydot_cons_notort[3] + ydot_flux[3]
            ]
        )
        return ydot

    def _single_pass_training(self,x):
        """
        Computes the RHS for the Dissipative Hamiltonian system.

        Args:
            x (jnp.ndarray): Input parameters [nu, r , phi, p_r, p_phi].

        Returns:
            jnp.ndarray: RHS of the Dissipative Hamiltonian system.
        """        
        nu = x[0]
        prims = x[1:]
        constants = self._set_eob_constants_3PN(nu)
        d_prims_dt = self._eom(0, prims, (nu, constants))
        return d_prims_dt

    def photon_effective_potential(self,r_grid,nu):
        constants = self._set_eob_constants_3PN(nu)
        a = jax.vmap(self._a_potential, in_axes=(0, None, None))(r_grid,nu,constants)
        return a/r_grid**2

    def particle_effective_potential(self,r_grid,j_grid,nu):
        constants = self._set_eob_constants_3PN(nu)
        a = jax.vmap(self._a_potential, in_axes=(0, None, None))(r_grid,nu,constants)
        return a*(1 + j_grid/r_grid**2)

    def __call__(self, x):
        """
        Compute the RHS for the Dissipative Hamiltonian system.

        Args:
            x (jnp.ndarray): batch of parameters of the form [nu, r, phi, p_r, p_phi]

        Returns:
            d_prims_dt (jnp.ndarray): RHS of the Dissipative Hamiltonian system.
        """
        return jax.vmap(self._single_pass_training, in_axes=(0))(x)

if __name__ == "__main__":
    eobnn = Neural_EOB(key=jax.random.PRNGKey(0),srate=4096)
    # test gradient through root finder
    @eqx.filter_jit
    def test_grad(model, x, y):
        y_pred = model(x)
        return jnp.mean(jnp.abs((y-y_pred)**2))

    print("Testing grad")
    x = jnp.load("seob_x_train_prelim.npy")
    y = jnp.load("seob_y_train_prelim.npy")
    try:
        val, grad = eqx.filter_value_and_grad(test_grad)(eobnn, x, y)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad)))
        print("Value:", val)
        print("Grad Norm:", grad_norm)
    except Exception as e:
        print("Error:", e)
