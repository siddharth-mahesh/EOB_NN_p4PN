"""
This file contains the Neural_EOB class,
which implements the non-spinning 3PN Effective One Body model
with 3.5PN circular radiation-reaction
and neural post-4PN terms.
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

from EOB_NN_p4PN.mlp import MLP
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
    eob3pn: EOB_3PN

    def __init__(
        self,
        key:jax.random.PRNGKey=jax.random.PRNGKey(42),
        srate:int=2000,
        hidden_dim_A:int=2,
        hidden_dim_D:int=2,
        hidden_dim_Q:int=2,
        hidden_dim_f:int=2,
        hidden_dim_delta:int=2,
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
        self.A_scale = 1.e-1
        self.D_scale = 1.e-1
        self.Q_scale = 1.e-1
        self.f_scale = 1.e-1
        self.delta_scale = 1.e-1
        self.eob3pn = EOB_3PN()
    def _strain(self, strain_qts, nu, constants):
        Omega = strain_qts[2] 
        f_nn = 1 + jnp.pow(Omega,7/2) * self.f_p4PN(jnp.array([Omega, nu]))
        delta_nn = jnp.exp(1j*jnp.pow(Omega,7/2) * self.delta_p4PN(jnp.array([Omega, nu])))
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
            1 + nu*self.A_scale*self.A_p4PN(neural_in)
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
        d_nn = 1 + jnp.pow(u,4) * self.D_p4PN(neural_in)
        d = self._pade_d(u, constants["d_2"], constants["d_3"]) 
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
        r, phi, p_r, p_phi = y
        u = 1 / r
        neural_q_in = jnp.array([nu, u, p_r])
        z_3 = constants["z_3"]
        a = self._a_potential(r, nu, constants)
        d = self._d_potential(r, nu, constants)
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
        return h_real
    
    def _lr_condition(self,r,params):
        """
        Solve for the LR condition for given nu and constants.

        Args:
            r (float): Radial position
            params (tuple): Parameters given by (nu, constants)

        Returns:
            float: LR condition
        """
        nu, constants = params
        #r_safe = jnp.maximum(r, 1.)
        r_safe = r
        # invert the sign on the photon effective potential to avoid any stable photon orbits
        photon_eff = lambda r_val: -self._a_potential(r_val, nu, constants) / r_val**2
        lr_condition = jax.grad(photon_eff)(r_safe)
        return lr_condition

    def _eom(self, t, y, args):
        """
        The equations of motion for the EOB model.

        Args:
            t (float): Time.
            y (jnp.ndarray): Canonical variables [r, phi, p_rstar, p_phi].
            args (tuple): Additional parameters (nu, r_ISCO, constants).

        Returns:
            jnp.ndarray: Equations of motion.
        """
        nu, _, constants = args
        num_coords = 2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords, num_coords)), jnp.eye(num_coords)],
                [-jnp.eye(num_coords), jnp.zeros((num_coords, num_coords))],
            ]
        )
        h , d_h_real = jax.value_and_grad(self._hamiltonian, argnums=0)(y, nu, constants)
        omega = d_h_real[3]  # omega = d_h_real/d_p_phi
        strain_qts = jnp.array([y[1], h*nu, omega])
        flux = self._flux(strain_qts, nu, constants)
        ydot = symplectic_map @ d_h_real + jnp.array([0.0, 0.0, 0.0, flux])
        return ydot

    def _j(self, r, nu, constants):
        """
        Compute the circular orbit angular momentum.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Circular orbit angular momentum
        """
        r3 = r * r * r
        a = self._a_potential(r, nu, constants)
        da_dr = jax.grad(self._a_potential, argnums=0)(r, nu, constants)
        j = jnp.sqrt(r3 * da_dr / (2 * a - r * da_dr))
        return j

    def _event_fn(self, t, y, args, **kwargs):
        """
        Compute the event function for termination at ISCO.

        Args:
            t (float): Time.
            y (jnp.ndarray): Canonical variables [r, phi, p_rstar, p_phi].
            args (tuple): Additional parameters (nu, r_ISCO, constants).

        Returns:
            float: ISCO event function.
        """
        _, r_fin, _ = args
        r, _, _, _ = y
        return r - r_fin

    def _dynamics(self, y0, nu, constants, dt=0.1):
        """
        Evolve the EOB dynamics.

        Args:
            y0 (jnp.ndarray): Initial conditions [r, phi, p_r, p_phi]
            nu (jnp.ndarray): Symmetric mass ratio
            constants (dict): Dictionary of constants
            dt (float): Output time step

        Returns:
            jnp.ndarray: Trajectory of the system
        """
        r_LR = jax.lax.stop_gradient(optimistix.root_find(
            self._lr_condition, optimistix.Newton(1e-8, 1e-8), 3.0, (nu, constants)
        ).value)
        params = (nu, r_LR, constants)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._eom),
            diffrax.Euler(),
            t0=0,
            t1=5000.0,
            dt0=dt,
            y0=y0,
            args=params,
            event=diffrax.Event(
                self._event_fn, optimistix.Newton(1e-4, 1e-4, optimistix.rms_norm)
            ),
            saveat=diffrax.SaveAt(t0=True, t1=True, dense=True),
            max_steps=100000,
            throw=False,
        )
        # restrict to 2000M before merger
        t_fin = jax.lax.stop_gradient(sol.ts[-1])
        times = jnp.linspace(0, t_fin - .1, self.srate)
        trajectory = jax.vmap(sol.evaluate, in_axes=0)(times)
        return times, trajectory

    def _strain_from_dynamics(self, trajectory, nu, constants):
        """
        Compute the GW strain given the trajectory

        Args:
            trajectory (jnp.ndarray): trajectory of the system
            nu (float): symmetric mass ratio
            constants (Dict): dictionary of EOB constants

        Returns:
            strain (complex): Complex GW strain.
        """
        h , dh_real = jax.vmap(jax.value_and_grad(self._hamiltonian, argnums=0), in_axes=(0, None, None))(trajectory, nu, constants)
        strain_qts = jnp.hstack([
            trajectory[:,1].reshape(trajectory.shape[0],1),
            h.reshape(h.shape[0],1)*nu,
            dh_real[:,3].reshape(dh_real.shape[0],1)
        ])
        return jax.vmap(self._strain, in_axes=(0, None, None))(
            strain_qts, nu, constants
        )
    
    def _single_pass_training(self,x):
        """
        Compute the GW strain given the parameters.
        This differs from _single_pass in that:
        1. It uses a pure 3PN rootfinder for initial conditions, so r0 aligns with omega_0.
        2. It does not use a root finder for final conditions, opts for fiducial t_fin = 2000M
        3. It uses the adjoint method for dynamics.

        Args:
        """        
        nu = x[0]
        constants = self._set_eob_constants_3PN(nu)
        ics_0 = self.eob3pn._initial_conditions(x)
        times_3pn, trajectory_3pn = self.eob3pn._dynamics(ics_0, nu, constants)
        t_fin_3pn = times_3pn[-1]
        t_start_nn = jnp.maximum(0.0, t_fin_3pn - 1000.0)
        ics_1000M = jnp.vstack([jnp.interp(t_start_nn, times_3pn, trajectory_3pn[:, i]) for i in range(4)]).flatten()
        ics = jnp.where(t_fin_3pn < 1000.0, ics_0, ics_1000M)
        ics = jax.lax.stop_gradient(ics)
        times, trajectory = self._dynamics(ics, nu, constants)
        times_stack = jnp.reshape(times, (times.shape[0], 1))
        strain = self._strain_from_dynamics(trajectory, nu, constants)
        strain_stack = jnp.reshape(strain, (strain.shape[0], 1))
        return jnp.hstack((times_stack, strain_stack), dtype=jnp.complex128)
    
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
        Compute the GW strain for a given batch of parameters

        Args:
            x (jnp.ndarray): batch of parameters of the form [nu]

        Returns:
            times (jnp.ndarray): times
            strain (complex): Complex GW strain.
        """
        return jax.vmap(self._single_pass_training, in_axes=(0))(x)

if __name__ == "__main__":
    from EOB_NN_p4PN.EOB_3PN.eob3pn import EOB
    eobnn = Neural_EOB(key=jax.random.PRNGKey(0),srate=4096)
    eob3pn = EOB()
    key = jax.random.PRNGKey(12)
    x_sxs = jnp.load("x_sxs_1em4.npy") 
    y_sxs = jnp.load("y_sxs_1em4.npy")
    strain_series = eobnn(x_sxs)
    print(f"strain_series.shape: {strain_series.shape}")
    strain_series_3pn = eob3pn(x_sxs)
    idx = jax.random.randint(key, 1, 0, x_sxs.shape[0] - 1)[0]
    times_3pn = strain_series_3pn[idx, :, 0] - strain_series_3pn[idx, -1, 0]
    strain_3pn_unprocessed = strain_series_3pn[idx, :, 1]
    times_sxs = y_sxs[idx, :, 0] - y_sxs[idx, -1, 0]
    strain_sxs_unprocessed = y_sxs[idx, :, 1]
    times = strain_series[idx, :, 0] - strain_series[idx, -1, 0]
    strain_unprocessed = strain_series[idx, :, 1]
    
    # zero phase at merger
    strain = strain_unprocessed*jnp.exp(-1j*jnp.angle(strain_unprocessed)[-1])
    print(jnp.unwrap(jnp.angle(strain))[-1])
    strain_3pn = strain_3pn_unprocessed*jnp.exp(-1j*jnp.angle(strain_3pn_unprocessed)[-1])
    print(jnp.unwrap(jnp.angle(strain_3pn))[-1])
    strain_sxs = strain_sxs_unprocessed*jnp.exp(-1j*jnp.angle(strain_sxs_unprocessed)[-1])
    print(jnp.unwrap(jnp.angle(strain_sxs))[-1])
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
    fig.subplots_adjust(hspace=0)
    ax[0].plot(jnp.real(times), jnp.abs(strain), label="Neural")
    ax[0].plot(jnp.real(times_3pn), jnp.abs(strain_3pn), label="3PN", linestyle="dashed")
    ax[0].plot(jnp.real(times_sxs), jnp.abs(strain_sxs), label="SXS", linestyle="dotted")
    ax[1].plot(jnp.real(times), jnp.unwrap(jnp.angle(strain)) - jnp.unwrap(jnp.angle(strain))[-1], label="Neural")
    ax[1].plot(
        jnp.real(times_3pn),
        jnp.unwrap(jnp.angle(strain_3pn)) - jnp.unwrap(jnp.angle(strain_3pn))[-1],
        label="3PN",
        linestyle="dashed",
    )
    ax[1].plot(jnp.real(times_sxs), jnp.unwrap(jnp.angle(strain_sxs)) - jnp.unwrap(jnp.angle(strain_sxs))[-1], label="SXS", linestyle="dotted")
    ax[0].set_ylabel(r"$A$")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(r"$\phi$")
    ax[1].legend()
    plt.savefig("eob_nnp4pn.png")

    from jax.scipy.integrate import trapezoid
    t_eval = jnp.linspace(-500,0,1024)
    dEdt_sxs = jnp.interp(t_eval,jnp.real(times_sxs),jnp.abs(jnp.gradient(strain_sxs, jnp.real(times_sxs)))**2)
    dEdt_nn = jnp.interp(t_eval,jnp.real(times),jnp.abs(jnp.gradient(strain, jnp.real(times)))**2)
    dEdt_3pn = jnp.interp(t_eval,jnp.real(times_3pn),jnp.abs(jnp.gradient(strain_3pn, jnp.real(times_3pn)))**2)
    phi_model = jnp.interp(t_eval,jnp.real(times),jnp.unwrap(jnp.angle(strain)) - jnp.unwrap(jnp.angle(strain))[-1])
    phi_sxs = jnp.interp(t_eval,jnp.real(times_sxs),jnp.unwrap(jnp.angle(strain_sxs)) - jnp.unwrap(jnp.angle(strain_sxs))[-1])
    phi_3pn = jnp.interp(t_eval,jnp.real(times_3pn),jnp.unwrap(jnp.angle(strain_3pn)) - jnp.unwrap(jnp.angle(strain_3pn))[-1])
    delta_phi_model_sxs = phi_model - phi_sxs
    delta_phi_model_3pn = phi_3pn - phi_model
    delta_phi_3pn_sxs = phi_3pn - phi_sxs
    Delta_phi_model_sxs = trapezoid(delta_phi_model_sxs**2,t_eval)
    Delta_phi_model_3pn = trapezoid(delta_phi_model_3pn**2,t_eval)
    Delta_phi_3pn_sxs = trapezoid(delta_phi_3pn_sxs**2,t_eval)
    Delta_E_GW_model_sxs = trapezoid((dEdt_nn-dEdt_sxs)**2,t_eval)
    Delta_E_GW_model_3pn = trapezoid((dEdt_nn-dEdt_3pn)**2,t_eval)
    Delta_E_GW_3pn_sxs = trapezoid((dEdt_3pn-dEdt_sxs)**2,t_eval)
    print(f"""
For the last 500M of the waveform, 
the excess radiated energy is:
Neural-SXS: {Delta_E_GW_model_sxs:.2e}
Neural-3PN: {Delta_E_GW_model_3pn:.2e}
3PN-SXS: {Delta_E_GW_3pn_sxs:.2e}

and accumulated phase difference (root squared error) is:
Neural-SXS: {Delta_phi_model_sxs:.2e}
Neural-3PN: {Delta_phi_model_3pn:.2e}
3PN-SXS: {Delta_phi_3pn_sxs:.2e}
""")
    
    fig,ax = plt.subplots(2,1,sharex=True, figsize=(8,10))
    fig.subplots_adjust(hspace=0)
    ax[0].plot(t_eval,dEdt_nn,label="Neural")
    ax[0].plot(t_eval,dEdt_3pn,label="3PN",linestyle="dashed")
    ax[0].plot(t_eval,dEdt_sxs,label="SXS",linestyle="dotted")
    ax[0].set_ylabel(r"$P_{\mathrm{GW}}$")
    ax[0].legend()
    ax[1].plot(t_eval,delta_phi_model_sxs,label="Neural-SXS")
    ax[1].plot(t_eval,delta_phi_model_3pn,label="Neural-3PN",linestyle="dashed")
    ax[1].plot(t_eval,delta_phi_3pn_sxs,label="3PN-SXS",linestyle="dotted")
    ax[1].set_ylabel(r"$\Delta \phi$")
    ax[1].legend()
    plt.savefig("eob_nnp4pn_energy_dephasing.png")

    rgrid = jnp.linspace(0.1, 10, 100)
    agrid_nn = eobnn.photon_effective_potential(rgrid, 0.25)
    agrid_3pn = eob3pn.photon_effective_potential(rgrid, 0.25)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(rgrid, agrid_nn, label="Neural")
    ax.plot(rgrid, agrid_3pn, label="3PN", linestyle="dashed")
    ax.set_xlabel("r")
    ax.set_ylabel("A")
    ax.legend()
    plt.savefig("eob_nnp4pn_photon_effective_potential.png")