# dho_dhnn.py
"""
Dissipative Hamiltonian Neural Network (D-HNN) model for damped harmonic oscillator.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class DHNN_Model(Model):
    """
    Dissipative Hamiltonian Neural Network (D-HNN) model.
    """
    def __init__(self, input_dim=2, hidden_dim=200, num_layers=2):
        """
        Initialize the D-HNN model.

        Args:
            input_dim (int): The dimension of the input.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int): The number of hidden layers.
        """
        super(DHNN_Model, self).__init__()
        
        # Start with an input layer
        hnn_layers = [tf.keras.layers.InputLayer(input_shape=(input_dim,))]
        dnn_layers = [tf.keras.layers.InputLayer(input_shape=(1,))]
        # Add the specified number of hidden layers
        for _ in range(num_layers):
            hnn_layers.append(tf.keras.layers.Dense(hidden_dim, activation='tanh'))
            dnn_layers.append(tf.keras.layers.Dense(hidden_dim, activation='tanh'))
        hnn_layers.append(tf.keras.layers.Dense(1))
        dnn_layers.append(tf.keras.layers.Dense(1))
        
        self.hnn = tf.keras.Sequential(hnn_layers)
        self.dnn = tf.keras.Sequential(dnn_layers)

    @tf.function
    def call(self, z):
        """
        Defines the forward pass to compute the time derivatives (dq/dt, dp/dt).
        The forward pass is given by the dissipative Hamiltonian equations
        dq/dt = dH/dp + dD/dq
        dp/dt = -dH/dq + dD/dp

        Args:
            z (tf.Tensor): The state vector [q, p].

        Returns:
            tf.Tensor: The time derivatives [dq_dt, dp_dt].
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z)
            q , p = tf.split(z, 2, axis = -1)
            H = self.hnn(z)
            D = self.dnn(p)

        grad_H = tape.gradient(H, z)
        grad_D = tape.gradient(D, p)
        del tape

        dq_dt_H = grad_H[:, 1]
        dp_dt_H = -grad_H[:, 0]
        dp_dt_D = grad_D[:,0]


        dq_dt = dq_dt_H
        dp_dt = dp_dt_H + dp_dt_D

        return tf.stack([dq_dt, dp_dt], axis=-1)
    
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
        derivatives = self.call(z_tensor)
        dq_dt, dp_dt = tf.split(derivatives, 2, axis=-1)

        # Leapfrog update
        p_half = p + 0.5 * dt * dp_dt
        q_next = q + dt * p_half
        z_next_q = tf.concat([q_next, p_half], axis=-1)
        derivatives_next = self.call(z_next_q)
        dq_dt_next, dp_dt_next = tf.split(derivatives_next, 2, axis=-1)
        p_next = p_half + 0.5 * dt * dp_dt_next

        z_next = tf.concat([q_next, p_next], axis=-1)
        dz_next = self.call(z_next)
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
        dz0 = self.call(z0)
        derivatives = derivatives.write(0, dz0)
        z_next = z0
        for i in range(t_points - 1):
            z_next , dz_next = self._leapfrog_step(z_next, dt)
            trajectory = trajectory.write(i + 1, z_next)
            derivatives = derivatives.write(i + 1, dz_next)
        return trajectory.stack() , derivatives.stack()

if __name__=="__main__":
    dhnn = DHNN_Model()
    z0 = tf.constant([[1.0, 0.0]], dtype=tf.float32)
    dz0 = dhnn(z0)
    print(dz0)
    