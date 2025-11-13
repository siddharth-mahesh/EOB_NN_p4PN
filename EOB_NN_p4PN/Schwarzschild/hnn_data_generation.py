# hnn_data_generation.py
"""
    This file contains the HamiltonianDataGenerator class, which is used to generate
    trajectories for relativistic orbits in Schwarzschild spacetime.
"""

# Core libraries
import jax
import jax.numpy as jnp
import diffrax

# --- 1. Class-based Data Generator (Debugged) ---
class HamiltonianDataGenerator:
    """
    A class to generate trajectories for relativistic orbits in Schwarzschild spacetime.
    """
    def __init__(self):
        """
        Initialize the data generator with system-specific parameters.
        """
        self.num_points_per_traj = 128

    def _hamiltonian(self,y):
        """
        Compute the Hamiltonian.

        Args:
            y (jnp.ndarray): The state vector [r, phi, p_r, p_phi].

        Returns:
            float: The Hamiltonian.
        """
        r , _ , p_r , p_phi = y
        a_potential = 1 - 2/r
        energy_sq = a_potential * (1 + a_potential * p_r**2 + p_phi**2 / r**2)
        energy = jnp.sqrt(energy_sq)
        return energy.squeeze()

    def _dynamics(self, t, y, args=None):
        """
        Compute the time derivatives using Hamilton's equations.

        Args:
            t (float): The time at which to compute the derivatives.
            y (jnp.ndarray): The state vector [r, phi, p_r, p_phi].

        Returns:
            jnp.ndarray: The time derivatives [dr/dt, dphi/dt, dp_r/dt, dp_phi/dt].
        """
        dH = jax.grad(self._hamiltonian, argnums=0)(y)
        num_coords = y.shape[0]//2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords, num_coords)), jnp.eye(num_coords)],
                [-jnp.eye(num_coords), jnp.zeros((num_coords, num_coords))],
            ]
        )
        return symplectic_map @ dH
    
    def _trajectory(self, y0):
        """
        Compute the trajectory for a given initial condition.

        Args:
            y0 (jnp.ndarray): The initial state vector [r, phi, p_r, p_phi].

        Returns:
            jnp.ndarray: The trajectory at the given time points.
        """
        # fiducial orbital timescale T = 2*pi*r**(3/2)
        t_span = (0.0, 2*jnp.pi*y0[0]**(3/2))
        ts = jnp.linspace(t_span[0], t_span[1], self.num_points_per_traj)
        dt_initial = ts[1]/100
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._dynamics),
            diffrax.Tsit5(),
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt_initial,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-11),
        )
        return ts, sol.ys
    
    def _calculate_strain(self, z):
        """
        Compute the complex strain at the given point.

        Args:
            z (jnp.ndarray): The input vector [nu, r, phi, p_r, p_phi].

        Returns:
            float: The absolute value of the complex strain at the given time points.
        """
        r, phi, p_r, p_phi = z
        y = jnp.array([r, phi, p_r, p_phi])
        rdot, phidot, _ , _ = self._dynamics(0, y)      
        H_22 =  1 / r + r**2 * phidot**2 - rdot**2
        h22 = 4 * H_22 * jnp.exp(-2j * phi)
        return jnp.abs(h22)
    
    def _single_forward(self, z):
        r, phi, p_r, p_phi = z
        y = jnp.array([r, phi, p_r, p_phi])
        ts , ys = self._trajectory(y)
        zs = ys
        h22 = jax.vmap(self._calculate_strain, in_axes=0)(zs)
        return zs , h22

    def __call__(self, z0, noise_std = 0, noise_key = jax.random.PRNGKey(24)):
        """
        Generate the training data for given batch of inputs.

        Args:
            z0 (jnp.ndarray): The batch of input vectors [nu, r, phi, p_r, p_phi].
            noise_std (float): The standard deviation of the noise.
            noise_key (jnp.ndarray): The random key for noise generation.

        Returns:
            jnp.ndarray: The batch of input vectors [nu, r, phi, p_r, p_phi] from trajectory.
            jnp.ndarray: The batch of train amplitudes.
        """
        zs , h22 = jax.vmap(self._single_forward, in_axes=0)(z0)
        # rebatch zs (currently has shape (batch_size,num_points_per_trajectory))
        x_train = zs.reshape(z0.shape[0]*self.num_points_per_traj, 4)
        y_train = h22.reshape(z0.shape[0]*self.num_points_per_traj, 1)
        return x_train, y_train

if __name__ == '__main__':
    generator = HamiltonianDataGenerator()
    key = jax.random.PRNGKey(24)
    r_key, nu_key = jax.random.split(key,2)
    n_batch = 1
    rs = jax.random.uniform(r_key, (n_batch, 1), minval=7.0, maxval=10.0)
    js = jnp.sqrt(rs/(1 - 3/rs))
    z0 = jnp.concatenate([rs,jnp.zeros((n_batch,1)),jnp.zeros((n_batch,1)),js],axis=-1)
    x_train, y_train = generator(z0)
    print(x_train.shape)
    print(y_train.shape)
    import matplotlib.pyplot as plt
    r_train = x_train[:,0]
    phi_train = x_train[:,1]
    plt.scatter(r_train*jnp.cos(phi_train),r_train*jnp.sin(phi_train))
    plt.show()
