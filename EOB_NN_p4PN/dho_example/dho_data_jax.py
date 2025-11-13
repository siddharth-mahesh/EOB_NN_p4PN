# dho_data.py
"""
Code to generate training data for the damped harmonic oscillator.
"""
import diffrax
import jax
import jax.numpy as jnp
import numpy as np


class DampedHarmonicOscillator:
    """
    Class to generate training data for the damped harmonic oscillator.
    """

    def __init__(self):
        """
        Initialize the damped harmonic oscillator.
        """
        self.t_span = (0.0, 10.0)
        self.num_points_per_traj = 100

    def _hamiltonian(self, y):
        """
        Compute the Hamiltonian.

        Args:
            y (jax.numpy.ndarray): The state vector [q, p].
            damping (float): The damping coefficient 'b'.

        Returns:
            jax.numpy.ndarray: The Hamiltonian.
        """
        q, p = y
        hamiltonian = 0.5 * p**2 + 0.5 * q**2
        return hamiltonian

    def _dissipative_potential(self, y, damping):
        """
        Compute the dissipative potential.

        Args:
            y (jax.numpy.ndarray): The state vector [q, p].
            damping (float): The damping coefficient 'b'.

        Returns:
            jax.numpy.ndarray: The dissipative potential.
        """
        _, p = y
        dissipative_potential = -0.5 * damping * p**2
        return dissipative_potential

    def _single_forward(
        self,
        x,
    ):
        """
        Compute the right-hand side for the damped harmonic oscillator.

        Args:
            x (jax.numpy.ndarray): The input vector [q, p, b].

        Returns:
            jax.numpy.ndarray: The right-hand side of the damped harmonic oscillator.
        """
        num_canonical_variables = x.shape[0] - 1
        half_size = num_canonical_variables // 2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((half_size, half_size)), jnp.eye(half_size)],
                [-jnp.eye(half_size), jnp.zeros((half_size, half_size))],
            ]
        )
        b = x[-1]
        grad_hamiltonian = jax.grad(self._hamiltonian, argnums=0)(x[:-1])
        grad_dissipative_potential = jax.jacobian(self._dissipative_potential, argnums=0)(x[:-1], b)
        dz = symplectic_map @ grad_hamiltonian + grad_dissipative_potential
        return dz

    def _ode_wrapper(
        self,
        t, # pylint: disable=unused-argument
        y,
        args
    ):
        return self._single_forward(jnp.hstack([y, args["damping"]]))

    # Define a function to solve for a single trajectory
    def _solve_single(self, x):
        y0 = x[:-1]
        b = x[-1]
        t_start, t_end = self.t_span
        # limit to 3 e-folding times for quality of data
        t_end = 3 * 2.0 / b
        ts = jnp.linspace(t_start, t_end, self.num_points_per_traj)
        dt_initial = ts[1] - ts[0]
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._ode_wrapper),
            diffrax.Tsit5(),
            t0=t_start,
            t1=t_end,
            dt0=dt_initial,
            y0=y0,
            args={"damping": b},
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-9),
        )
        return ts, jnp.hstack([sol.ys, jnp.tile(b, (self.num_points_per_traj, 1))])

    def _solve(self, x):
        return jax.vmap(self._solve_single, in_axes=0)(x)

    def _forward(self, x):
        return jax.vmap(self._single_forward, in_axes=0)(x)

    def __call__(self, x0, rhs=True):
        """
        Generate the training data.
        The class takes in a batch of phase space points
        and returns either the right-hand side of the damped harmonic oscillator (rhs=True)
        or the Hamiltonian and dissipative potential (rhs=False).

        Args:
            x0 (jax.numpy.ndarray): The initial state vector [q, p, b].
            rhs (bool): Whether to return the right-hand side of the damped harmonic oscillator.
        Returns:
            tuple: The training data (x_data, y_data). (rhs = True)
            tuple: The trajectory data (times, trajectories). (rhs = False)
        """
        times, trajectories = self._solve(x0)
        if not rhs:
            return times, trajectories
        num_points = self.num_points_per_traj * x0.shape[0]
        # Stack the trajectories and the damping parameter
        x_data = trajectories.reshape(num_points, 3)
        y_data = self._forward(x_data)
        return x_data, y_data

def generate_training_data(num_trajectories, seed, val_split):
    key = jax.random.PRNGKey(seed)
    key_prim, key_damp = jax.random.split(key,2)
    prims = jax.random.uniform(key_prim, (num_trajectories, 2), minval=-1, maxval=1)
    dampings = 10 ** jax.random.uniform(
        key_damp, (num_trajectories, 1), minval=jnp.log10(0.1), maxval=jnp.log10(2)
    )
    x0s = jnp.hstack([prims, dampings])
    data_class = DampedHarmonicOscillator()
    x , y = data_class(x0s, rhs=True)
    num_train = int(x.shape[0] * (1 - val_split))
    x_train = x[:num_train]
    y_train = y[:num_train]
    x_val = x[num_train:]
    y_val = y[num_train:]
    return (x_train, y_train), (x_val, y_val)

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = generate_training_data(100, 42, 0.2)
    np.savetxt("x_train.dat", np.asarray(x_train))
    np.savetxt("y_train.dat", np.asarray(y_train))
    np.savetxt("x_val.dat", np.asarray(x_val))
    np.savetxt("y_val.dat", np.asarray(y_val))
