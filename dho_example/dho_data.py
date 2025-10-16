# dho_data.py
"""
    Code to generate training data for the damped harmonic oscillator.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DampedHarmonicOscillator:
    """
    Class to generate training data for the damped harmonic oscillator.
    """
    def __init__(self, b=0.5):
        """
        Initialize the damped harmonic oscillator.

        Args:
            b (float): Damping coefficient.
        """
        self.b = b
    def _hamiltonian_and_dissipative_potential(self, z):
        """
        Compute the Hamiltonian and dissipative potential.

        Args:
            z (tf.Tensor): The state vector [q, p].

        Returns:
            tuple: The Hamiltonian and dissipative potential.
        """
        q, p = tf.split(z, 2, axis=-1)
        #H = 0.5 * p**2 + (1 - tf.math.cos(q))*0.5
        H = 0.5 * p**2 + 0.5 * q**2
        D = -0.5 * self.b * p**2
        return H, D
    def _dynamics(self, z):
        """
        Compute the right-hand side for the damped harmonic oscillator.

        Args:
            z (tf.Tensor): The state vector [q, p].

        Returns:
            tf.Tensor: The right-hand side of the damped harmonic oscillator.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z)
            q, p = tf.split(z, 2, axis=-1)
            #H = 0.5 * p**2 + (1 - tf.math.cos(q))*0.5
            H = 0.5 * p**2 + 0.5 * q**2
            D = -0.5 * self.b * p**2
        
        dH_dp = tape.gradient(H, p)
        dH_dq = tape.gradient(H, q)
        dD_dp = tape.gradient(D, p)
        del tape

        dq_dt = dH_dp
        dp_dt = -dH_dq + dD_dp
        return tf.concat([dq_dt, dp_dt], axis=-1)
    def _leapfrog_step(self, z, dt):
        """
        Perform a single leapfrog step.
        The leapfrog method is given by the kick-drift-kick scheme in
        https://en.wikipedia.org/wiki/Leapfrog_integration

        Args:
            z (tf.Tensor): The state vector [q, p].
            dt (float): The time step.

        Returns:
            tf.Tensor: The updated state vector [q, p].
            tf.Tensor: The updated derivatives [dq_dt, dp_dt].
        """
        q, p = tf.split(z, 2, axis=-1)
        z_tensor = tf.concat([q, p], axis=-1)

        # Get the derivatives from the model
        derivatives = self._dynamics(z_tensor)
        dq_dt, dp_dt = tf.split(derivatives, 2, axis=-1)

        # Leapfrog update
        p_half = p + 0.5 * dt * dp_dt
        q_next = q + dt * p_half
        z_next_q = tf.concat([q_next, p_half], axis=-1)
        derivatives_next = self._dynamics(z_next_q)
        dq_dt_next, dp_dt_next = tf.split(derivatives_next, 2, axis=-1)
        p_next = p_half + 0.5 * dt * dp_dt_next

        z_next = tf.concat([q_next, p_next], axis=-1)
        dz_next = self._dynamics(z_next)
        return z_next , dz_next
    def _leapfrog_integrator(self, z0, t_span, t_points):
        """
        Integrate the system using the leapfrog integrator.

        Args:
            z0 (tf.Tensor): The initial state vector [q, p].
            t_span (tuple): The time span (t_start, t_end).
            t_points (int): The number of time points.

        Returns:
            tf.Tensor: The trajectory of the system.
            tf.Tensor: The derivatives of the system.
        """
        dt = (t_span[1] - t_span[0]) / (t_points - 1)
        trajectory = tf.TensorArray(tf.float32, size=t_points)
        derivatives = tf.TensorArray(tf.float32, size=t_points)
        trajectory = trajectory.write(0, z0)
        dz0 = self._dynamics(z0)
        derivatives = derivatives.write(0, dz0)
        z_next = z0
        for i in range(t_points - 1):
            z_next , dz_next = self._leapfrog_step(z_next, dt)
            trajectory = trajectory.write(i + 1, z_next)
            derivatives = derivatives.write(i + 1, dz_next)
        return trajectory.stack() , derivatives.stack()
    
    def __call__(self, z0 , t_span , t_points, num_samples=1):
        """
        Generate the training data.
        The class takes in arbitrary initial conditions
        but provides an additional sampling of the phase space
        by integrating the damped harmonic oscillator for a fixed time.

        If inputs are given as a batch of initial conditions with the shape (None, 2),
        then the output of the leapfrog integrator will be a batch of trajectories and derivatives with the shape (t_points, None, 2).
        The output of the leapfrog integrator is then reshaped into the form of (t_points * None, 2)
        to be used as training data.


        Args:
            z0 (tf.Tensor): The initial state vector [q, p].
            t_span (tuple): The time span (t_start, t_end).
            t_points (int): The number of time points.
            num_samples (int): The number of samples.

        Returns:
            tuple: The training data (x_data, y_data).
        """
        batched_trajectory , batched_derivatives = self._leapfrog_integrator(z0, t_span, t_points)
        # use every t_points//num_samples number of trajectories
        batched_trajectory = batched_trajectory[::t_points//num_samples]
        batched_derivatives = batched_derivatives[::t_points//num_samples]
        
        
        
        # batched trajectory has shape (t_points, None, 2)
        # training data has shape (t_points * None, 2)
        x_data = tf.reshape(batched_trajectory, (-1, 2))
        y_data = tf.reshape(batched_derivatives, (-1, 2))
        return x_data , y_data

if __name__ == '__main__':
    
    # span of phase space qs in range -1 to 1
    rng = np.random.default_rng()
    num_grid_pts = 20
    qs = rng.uniform(-2, 2, num_grid_pts)
    ps = rng.uniform(-2, 2, num_grid_pts)
    z0 = tf.constant([
        [qs[i], ps[i]] for i in range(num_grid_pts)
    ], dtype=tf.float32)
    b = 0.5
    damping_time_scale = 1/b
    t_span = [0, 10*damping_time_scale]
    t_points = 100
    data_class = DampedHarmonicOscillator(b)

    x_data , y_data = data_class(z0, t_span, t_points)
    trajectory , derivatives = data_class._leapfrog_integrator(z0, t_span, t_points)

    # --- 6. Visualization and Analysis ---

    # plot the phase space of a particular trajectory
    plt.figure(figsize=(10, 5))
    batch_idx = rng.integers(0 , num_grid_pts)
    plt.plot(trajectory[:, batch_idx, 0], trajectory[:, batch_idx, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('square')
    plt.title("Phase Space of a Particular Trajectory")
    plt.xlabel("Position (q)")
    plt.ylabel("Momentum (p)")
    plt.show()

    # plot the Hamiltonian and dissipative potential over the phase space
    fig , axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_grid_size = 500
    q = np.linspace(-2, 2, plot_grid_size)
    p = np.linspace(-2, 2, plot_grid_size)
    Q , P = np.meshgrid(q, p)
    grid_points = tf.stack([Q.flatten(), P.flatten()], axis=-1)
    potentials = data_class._hamiltonian_and_dissipative_potential(grid_points)
    H = potentials[0].numpy().reshape(plot_grid_size, plot_grid_size)
    D = potentials[1].numpy().reshape(plot_grid_size, plot_grid_size)

    contour1 = axes[0].contourf(Q, P, H)
    fig.colorbar(contour1, ax=axes[0])
    axes[0].set_title("Hamiltonian")
    axes[0].set_xlabel("Position (q)")
    axes[0].set_ylabel("Momentum (p)")
    
    contour2 = axes[1].contourf(Q, P, D)
    fig.colorbar(contour2, ax=axes[1])
    axes[1].set_title("Dissipative Potential")
    axes[1].set_xlabel("Position (q)")
    axes[1].set_ylabel("Momentum (p)")

    plt.tight_layout()
    plt.savefig('true_potentials.png')

    plt.show()
    
    
    
    
    
    