"""
This file contains the EOB_NN_p4PN class,
which implements the non-spinning 3PN Effective One Body model
with 3.5PN circular radiation-reaction
and neural post 4PN terms.
"""
I = 1j
import jax
import jax.numpy as jnp
from jax.numpy import log
from EOB_NN_p4PN.EOB.flux import _flux
from EOB_NN_p4PN.EOB.pade_1_3_a import _pade_1_3
from EOB_NN_p4PN.EOB.pade_0_3_d import _pade_0_3
from EOB_NN_p4PN.EOB.eob_constants_3pn import _set_eob_constants_3PN
from EOB_NN_p4PN.EOB.strain import _strain
# set jax to 64 bit precision
jax.config.update("jax_enable_x64", True)
import diffrax
import optimistix
import equinox as eqx

from EOB_NN_p4PN.mlp import MLP
from typing import Callable


class Neural_EOB(eqx.Module):
    """
    This class implements the non-spinning 3PN Effective One Body model
    with 3.5PN circular radiation-reaction
    and neural post 4PN terms.
    """
    conservative_order: int
    radiative_order: float
    A_p4PN: MLP
    D_p4PN: MLP
    Q_p4PN: MLP
    _flux: Callable
    _pade_a: Callable
    _pade_d: Callable
    _set_eob_constants_3PN: Callable
    _strain: Callable
    
    def __init__(self, key = jax.random.PRNGKey(42),hidden_dim_A = 16,hidden_dim_D = 16,hidden_dim_Q = 16,hidden_dim_h = 16):
        """
        Initialize the EOB class.
        """

        # model identifiers
        self.conservative_order = 3
        self.radiative_order = 3.5
        A_key, D_key, Q_key = jax.random.split(key, 3)
        self.A_p4PN = MLP(input_dim=2, output_dim='scalar', hidden_dim=hidden_dim_A, key=A_key)
        self.D_p4PN = MLP(input_dim=2, output_dim='scalar', hidden_dim=hidden_dim_D, key=D_key)
        self.Q_p4PN = MLP(input_dim=3, output_dim='scalar', hidden_dim=hidden_dim_Q, key=Q_key)
        self._flux = _flux
        self._pade_a = _pade_1_3   # Pade approximant for A potential
        self._pade_d = _pade_0_3   # Pade approximant for D potential
        self._set_eob_constants_3PN = _set_eob_constants_3PN
        self._strain = _strain

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
        neural_in = jnp.array([u,nu])
        a = self._pade_a(u, constants["a_1"], constants["a_3"], constants["a_4"]) * (1 + self.A_p4PN(neural_in))
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
        neural_in = jnp.array([u,nu])
        d = self._pade_d(u, constants["d_2"], constants["d_3"]) * (1 + self.D_p4PN(neural_in))
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
        r , phi , p_r , p_phi = y
        u = 1 / r
        neural_q_in = jnp.array([nu,u,p_r])
        z_3 = constants['z_3']
        a = self._a_potential(r,nu, constants)
        d = self._d_potential(r,nu, constants)
        q_p4pn = self.Q_p4PN(neural_q_in)
        h_real = jnp.sqrt(2*nu*(jnp.sqrt(a*(((p_phi)*(p_phi))*((u)*(u)) + ((p_r)*(p_r))*(a/d + ((p_r)*(p_r))*(((u)*(u))*z_3*(1 + q_p4pn))) + 1)) - 1) + 1)/nu
        return h_real
    
    def _c_potential(self, r, p_phi, nu, constants):
        """
        Compute the Hamiltonian C potential.
        The C potential is given by:
            C = lim_{p_r -> 0} (1/p_r * dH/dp_r)

        Args:
            r (float): Radial position.
            p_phi (float): Angular momentum.
            nu (float): Symmetric mass ratio.
            constants (dict): Dictionary of constants.

        Returns:
            float: Hamiltonian C potential.
        """
        u = 1 / (r + 1e-100)
        a = self._a_potential(r, nu, constants)
        d = self._d_potential(r, nu, constants)
        tmp0 = ((p_phi) * (p_phi)) * ((u) * (u))
        tmp2 = 2 * nu * (jnp.sqrt(a) * jnp.sqrt(tmp0 + 1) - 1)
        c_circ = jnp.pow(a, 3.0 / 2.0) / (d * jnp.sqrt(tmp0 * tmp2 + tmp0 + tmp2 + 1))
        return c_circ

    def _j(self, r,nu, constants):
        """
        Compute the circular orbit angular momentum.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Circular orbit angular momentum
        """
        r3 = r * r * r
        a = self._a_potential(r,nu, constants)
        da_dr = jax.grad(self._a_potential, argnums=0)(r,nu, constants)
        j = jnp.sqrt(r3 * da_dr / (2 * a - r * da_dr))
        return j

    def _h_circ(self, r, nu, constants):
        """
        Compute the circular orbit Hamiltonian.

        Args:
            r (float): Radial position
            nu (float): Symmetric mass ratio
            constants (dict): Dictionary of constants

        Returns:
            float: Circular orbit Hamiltonian
        """
        j = self._j(r,nu, constants)
        z_circ = jnp.array([r, 0.0, 0.0, j])
        h_circ = self._hamiltonian(z_circ, nu, constants)
        return h_circ

    def _pr_adiabatic(self, r, nu, omega_0, constants):
        """
        Compute the radial momentum in the adiabatic limit.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Radial momentum
        """
        j, dj_dr = jax.value_and_grad(self._j, argnums=0)(r, nu, constants)
        c = self._c_potential(r, j, nu, constants)
        v_w = jnp.pow(omega_0, 1 / 3)
        flux = self._flux(v_w, nu, constants)
        pr = flux / (c * dj_dr)
        return pr

    def _circular_orbit_condition(self, r, params):
        """
        Solve for the circular orbit condition for given nu and omega_0.

        Args:
            r (float): Radial position
            params (tuple): Parameters given by (nu, omega_0, constants)

        Returns:
            float: Circular orbit condition
        """
        nu, omega_0, constants = params
        j = self._j(r,nu, constants)
        y = jnp.array([r, 0.0, 0.0, j])
        d_h_real = jax.grad(self._hamiltonian, argnums=0)(y, nu, constants)
        phidot = d_h_real[3]  # omega = d_h_real/d_p_phi
        return phidot - omega_0

    def _isco_condition(self, r, params):
        """
        Solve for the ISCO condition for given nu and constants.

        Args:
            r (float): Radial position
            params (tuple): Parameters given by (nu, constants)

        Returns:
            float: ISCO condition
        """
        nu, constants = params
        dhdr = lambda r: jax.grad(self._h_circ)(r, nu, constants)
        d2h_dr2 = jax.grad(dhdr)(r)
        return d2h_dr2

    def _initial_conditions(self, x):
        """
        Find the initial conditions for the EOB Equations of motion.

        Args:
            x (jnp.ndarray): input data [nu, omega_0]

        Returns:
            jnp.ndarray: Initial conditions [r, phi, p_r, p_phi]
        """
        nu, omega_0 = x
        constants = self._set_eob_constants_3PN(nu)
        r0 = optimistix.root_find(self._circular_orbit_condition, optimistix.Newton(1e-12,1e-12), omega_0 ** (-2 / 3), (nu, omega_0, constants)).value
        pr0 = self._pr_adiabatic(r0, nu, omega_0, constants)
        pphi0 = self._j(r0,nu, constants)
        return jnp.array([r0, 0.0, pr0, pphi0])

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
        num_coords = len(y) // 2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords, num_coords)), jnp.eye(num_coords)],
                [-jnp.eye(num_coords), jnp.zeros((num_coords, num_coords))],
            ]
        )
        d_h_real = jax.grad(self._hamiltonian, argnums=0)(y, nu, constants)
        omega = d_h_real[3]  # omega = d_h_real/d_p_phi
        v = omega ** (1.0 / 3.0)
        flux = self._flux(v, nu, constants)
        ydot = symplectic_map @ d_h_real + jnp.array([0.,0.,0.,flux])
        return ydot

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
        _, r_ISCO, _ = args
        r, _, _, _ = y
        return r - r_ISCO

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
        r_ISCO = optimistix.root_find(self._isco_condition, optimistix.Newton(1e-12,1e-12), 6.0,(nu, constants)).value
        params = (nu, r_ISCO, constants)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._eom),
            diffrax.Dopri8(),
            t0=0,
            t1=jnp.inf,
            dt0=dt,
            y0=y0,
            args=params,
            stepsize_controller=diffrax.PIDController(rtol=1e-15, atol=1e-14),
            event=diffrax.Event(
                self._event_fn, optimistix.Newton(1e-5, 1e-5, optimistix.rms_norm)
            ),
            saveat=diffrax.SaveAt(t0=True, t1=True, dense=True),
        )
        times = jnp.linspace(0, sol.ts[-1], 3000)
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
        return jax.vmap(self._strain, in_axes=(None, 0, None, None))(
            self,trajectory, nu, constants
        )

    def _single_pass(self, x):
        """
        Compute the GW strain given the parameters

        Args:
            x (jnp.ndarray): parameters [nu, omega_0]

        Returns:
            times (jnp.ndarray): times
            strain (complex): Complex GW strain.
        """
        nu = x[0]
        constants = self._set_eob_constants_3PN(nu)
        ics = self._initial_conditions(x)
        times, trajectory = self._dynamics(ics, nu, constants)
        times_stack = jnp.reshape(times, (times.shape[0], 1))
        strain = self._strain_from_dynamics(trajectory, nu, constants)
        strain_stack = jnp.reshape(strain, (strain.shape[0], 1))
        return jnp.hstack((times_stack, strain_stack), dtype=jnp.complex128)
    
    def __call__(self, x):
        """
        Compute the GW strain for a given batch of parameters

        Args:
            x (jnp.ndarray): batch of parameters of the form [nu, omega_0]

        Returns:
            times (jnp.ndarray): times
            strain (complex): Complex GW strain.
        """
        return jax.jit(jax.vmap(self._single_pass, in_axes=(0)))(x)
        

if __name__ == "__main__":
    from EOB_NN_p4PN.EOB_3PN.eob3pn import EOB
    eobnn = Neural_EOB(key=jax.random.PRNGKey(50))
    eob3pn = EOB()
    key = jax.random.PRNGKey(0)
    idx_key , omega_key , nu_key = jax.random.split(key,3)
    num_cases = 2
    omegas = 0.01 + 0.005*jax.random.uniform(omega_key,(num_cases,1))
    nus = 0.1 + 0.15*jax.random.uniform(nu_key,(num_cases,1))
    x = jnp.hstack((nus,omegas))
    strain_series = eobnn(x)
    strain_series_3pn = eob3pn(x)
    idx = jax.random.randint(idx_key,1,0,num_cases-1)[0]
    times = strain_series[idx,:,0]
    strain = strain_series[idx,:,1]
    times_3pn = strain_series_3pn[idx,:,0]
    strain_3pn = strain_series_3pn[idx,:,1]

    import matplotlib.pyplot as plt
    fig , ax = plt.subplots(1,2)
    ax[0].plot(jnp.abs(times),jnp.real(strain),label = r'$h_+$')
    ax[0].plot(jnp.abs(times_3pn),jnp.real(strain_3pn),label = r'$h_+$')
    ax[1].plot(jnp.abs(times),-jnp.imag(strain),label = r'$h_\times$')
    ax[1].plot(jnp.abs(times_3pn),-jnp.imag(strain_3pn),label = r'$h_\times$')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Strain')
    ax[0].legend()
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Strain')
    ax[1].legend()
    plt.show()