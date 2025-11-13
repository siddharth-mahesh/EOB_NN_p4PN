# hnn_data_generation.py
"""
    This file contains the HamiltonianDataGenerator class, which is used to generate
    trajectories for relativistic orbits in Schwarzschild spacetime.
"""

# Core libraries
import tensorflow as tf
import numpy as np

# --- 1. Class-based Data Generator (Debugged) ---
class HamiltonianDataGenerator:
    """
    A class to generate trajectories for relativistic orbits in Schwarzschild spacetime.
    """
    def __init__(self, nu , omega_ref):
        """
        Initialize the data generator with system-specific parameters.

        Args:
            nu (float): The symmetric mass ratio of the binary system.
            omega_ref (float): The reference rotation rate for the symplectic integrator.
        """
        self.nu = nu
        self.omega_ref = omega_ref
    def _dynamics(self, t, y):
        """
        Compute the time derivatives using Hamilton's equations.

        Args:
            t (float): The time at which to compute the derivatives.
            y (tf.Tensor): The state vector [q, p].

        Returns:
            tf.Tensor: The time derivatives [dq/dt, dp/dt].
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(y)
            q, p = tf.split(y, 2, axis=-1)
            r = q[:, 0]
            p_r, p_phi = p[:, 0], p[:, 1]
            a = 1 - 2 / r
            energy_sq = a * (1 + a*p_r**2 + p_phi**2 / r**2)
            energy = tf.sqrt(energy_sq)
            
        dH_dp = tape.gradient(energy, p)
        dH_dq = tape.gradient(energy, q)
        del tape

        dq_dt = dH_dp
        dp_dt = -dH_dq
        return tf.concat([dq_dt, dp_dt], axis=-1)
    
    def _integrator_rk4(self,ts,x0):
        """
        Integrate the system using the Runge-Kutta 4 method.

        Args:
            ts (tf.Tensor): The time points at which to evaluate the trajectory.
            x0 (tf.Tensor): The initial state vector [q, p].

        Returns:
            tf.Tensor: The trajectory at the given time points.
            tf.Tensor: The derivatives at the given time points.
        """
        dt = ts[1] - ts[0]
        t_steps = len(ts)

        y_array = tf.TensorArray(tf.float32, size=t_steps)
        dydt_array = tf.TensorArray(tf.float32, size=t_steps) # Array to store derivatives

        y = x0
        y_array = y_array.write(0, y)
        dydt_array = dydt_array.write(0, self._dynamics(ts[0], y))

        for i in tf.range(t_steps - 1):

            # RK4 integration
            k1 = self._dynamics(ts[i], y)
            k2 = self._dynamics(ts[i] + dt/2, y + dt/2 * k1)
            k3 = self._dynamics(ts[i] + dt/2, y + dt/2 * k2)
            k4 = self._dynamics(ts[i] + dt, y + dt * k3)
            y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            dydt = self._dynamics(ts[i], y)
            y_array = y_array.write(i + 1, y)
            dydt_array = dydt_array.write(i + 1, dydt) # Store the derivative
        return y_array.stack(), dydt_array.stack()
    
    def _integrator_symplectic_Tao2016(self,ts,x0):
        """
        Integrate the system using a symplectic integrator for non-separable Hamiltonians.
        Second order accurate symplectic integrator given by Equation 2 of https://arxiv.org/pdf/1609.02212.

        Args:
            ts (tf.Tensor): The time points at which to evaluate the trajectory.
            x0 (tf.Tensor): The initial state vector [q, p].

        Returns:
            tf.Tensor: The trajectory at the given time points.
            tf.Tensor: The derivatives at the given time points.
        """
        dt = ts[1] - ts[0]
        t_steps = tf.shape(ts)[0]

        # Initialize the doubled phase space: (q, p) are physical, (x, y) are auxiliary
        q, p = tf.split(x0, 2, axis=-1)
        x, y = tf.identity(q), tf.identity(p)
        omega_ref = tf.cast(self.omega_ref, tf.float32)

        trajectory = tf.TensorArray(tf.float32, size=t_steps, element_shape=x0.shape)
        trajectory = trajectory.write(0, x0)
        derivatives = tf.TensorArray(tf.float32, size=t_steps, element_shape=x0.shape)
        derivatives = derivatives.write(0, self._dynamics(ts[0], x0))
        
        for i in tf.range(t_steps - 1):
            
            # --- STEP A: First half-step drift ---
            # This step is given by \phi^{dt/2}_{H_A}
            # H_A = H(q,y)
            state_A = tf.concat([q, y], axis=-1)
            dzdt = self._dynamics(ts[i],state_A)
            dqdt, dydt = tf.split(dzdt, 2, axis=-1)
            
            q = q
            p = p + (dt / 2.0) * dydt
            x = x + (dt / 2.0) * dqdt
            y = y

            # --- STEP B: Second half-step drift ---
            # This step is given by \phi^{dt/2}_{H_B}
            # H_B = H(x,p)

            state_B = tf.concat([x, p], axis=-1)
            dzdt = self._dynamics(ts[i],state_B)
            dxdt, dpdt = tf.split(dzdt, 2, axis=-1)

            q = q + (dt / 2.0) * dxdt
            p = p
            x = x
            y = y + (dt / 2.0) * dpdt

            
            # --- STEP C: Rotation in doubled phase space ---
            # This step is given by \phi^{dt}_{\omega H_C}
            # H_C = 0.5(||q - x|| + ||p - y||)
            Rs = tf.sin(2.0 * omega_ref * dt)
            Rc = tf.cos(2.0 * omega_ref * dt)
            R11 = Rc
            R12 = Rs
            R21 = -Rs
            R22 = Rc
            q_plus_x = q + x
            q_minus_x = q - x
            p_plus_y = p + y
            p_minus_y = p - y
            
            rot_transform_1 = R11*q_minus_x + R12*p_minus_y
            rot_transform_2 = R21*q_minus_x + R22*p_minus_y

            q = 0.5 * (q_plus_x + rot_transform_1)
            p = 0.5 * (p_plus_y + rot_transform_2)
            x = 0.5 * (q_plus_x - rot_transform_1)
            y = 0.5 * (p_plus_y - rot_transform_2)

            # --- STEP D: Third half-step drift ---
            # This step is structurally identical to Step B.
            stateB = tf.concat([x, p], axis=-1)
            dzdt = self._dynamics(ts[i],stateB)
            dxdt, dpdt = tf.split(dzdt, 2, axis=-1)
            
            q = q + (dt / 2.0) * dxdt
            p = p
            x = x
            y = y + (dt / 2.0) * dpdt

            # --- STEP E:(Final step) Fourth half-step drift ---
            # This step is structurally identical to Step A.
            stateA = tf.concat([q, y], axis=-1)
            dzdt = self._dynamics(ts[i],stateA)
            dqdt, dydt = tf.split(dzdt, 2, axis=-1)
            
            q = q
            p = p + (dt / 2.0) * dydt
            x = x + (dt / 2.0) * dqdt
            y = y


            # DEBUGGING: This will raise an error if any value becomes NaN or Inf.
            tf.debugging.check_numerics(q, "q became non-numeric")
            tf.debugging.check_numerics(p, "p became non-numeric")

            # Store the physical part of the trajectory for this time step.
            z = tf.concat([q, p], axis=-1)
            dzdt = self._dynamics(ts[i], z)
            trajectory = trajectory.write(i+1, z)
            derivatives = derivatives.write(i+1, dzdt)
       
        return trajectory.stack(), derivatives.stack()

    def _calculate_strain(self, ys, dydts):
        """
        Compute the complex strain at the given time points.

        Args:
            ys (tf.Tensor): The trajectory at the given time points.
            dydts (tf.Tensor): The derivatives at the given time points.

        Returns:
            tf.Tensor: The complex strain at the given time points.
        """
        q, _ = tf.split(ys, 2, axis=-1)
        dqdt, _ = tf.split(dydts, 2, axis=-1)
        
        r, phi = q[..., 0], q[..., 1]
        rdot, phidot = dqdt[..., 0], dqdt[..., 1]
        
        H_22 = tf.cast(1 / r + r**2 * phidot**2 - rdot**2, tf.complex64) + 2j * tf.cast(self.nu * rdot * phidot, tf.complex64)
        h22 = 4 *self.nu * H_22 * tf.exp(-2j * tf.cast(phi, tf.complex64))

        return h22

    def generate_data(self, t_span, t_points, x0, noise_std, return_amp_phase=False, out_derivatives=False):
        """
        Generate a trajectory and the corresponding gravitational waveform strain.

        Args:
            t_span (tuple): The time span (start, end) for the trajectory.
            t_points (int): The number of time points to generate.
            x0 (tf.Tensor): The initial state vector [q, p].
            noise_std (float): The standard deviation of the noise.
            return_amp_phase (bool): Whether to return the amplitude and phase of the strain.
            derivatives (bool): Whether to return the derivatives of the trajectory.

        Returns:
            np.ndarray: The time points at which the trajectory is evaluated.
            np.ndarray: The trajectory at the given time points.
            np.ndarray: The complex strain at the given time points OR the amplitude and phase of the strain at the given time points (if return_amp_phase is True).
            np.ndarray: The derivatives of the trajectory at the given time points (if derivatives is True).
        """
        t_eval = np.linspace(t_span[0], t_span[1], t_points, dtype=np.float32)
        
        print("Generating trajectory...")
        # 1. Integrate to get BOTH trajectory and derivatives
        #trajectory, derivatives = self._integrator_rk4(t_eval, x0)
        trajectory, derivatives = self._integrator_symplectic_Tao2016(t_eval, x0)
        print("Calculating complex strain...")
        # 2. Calculate the complex strain using both trajectory and derivatives
        h22_complex = self._calculate_strain(trajectory, derivatives)
        print("Adding noise...")
        # 5. Add noise (optional)
        if noise_std > 0:
            h22_complex += tf.random.normal(shape=h22_complex.shape, stddev=noise_std)
        
        if return_amp_phase:
            print("Extracting amplitude and wrapped phase...")
            # 3. Extract amplitude and wrapped phase
            amplitude = tf.abs(h22_complex)
            wrapped_phase = tf.math.angle(h22_complex)
        
            print("Unwrapping phase...")
            # 4. Unwrap the phase using NumPy
            unwrapped_phase_np = np.unwrap(wrapped_phase.numpy(), axis=0)
            unwrapped_phase = tf.convert_to_tensor(unwrapped_phase_np, dtype=tf.float32)
                
            print("Stacking amplitude and unwrapped phase...")
            # 6. Stack amplitude and unwrapped phase into the final output tensor
            h_amp_phase = tf.stack([amplitude, unwrapped_phase], axis=-1)
        
        h22_to_return = h22_complex if not return_amp_phase else h_amp_phase
        print("Returning data...")
        if out_derivatives:
            return t_eval, trajectory, h22_to_return, derivatives
        return t_eval, trajectory, h22_to_return

# --- 2. Example Usage (Unchanged) ---
if __name__ == '__main__':
    # Define simulation parameters
    noise_level = 0.0
    mass_ratio = 1e-5
    sym_mass_ratio = mass_ratio/(1 + mass_ratio)**2
    l_isco = 12
    r_1 = 20
    l_1 = np.sqrt(r_1**2/(r_1 - 3))
    r_2 = 22
    l_2 = np.sqrt(r_2**2/(r_2 - 3))
    t_start = 0.0
    t_p = 2*np.pi*(r_2**(1.5))
    t_end = 1*t_p
    print(f"t_p: {t_p}")
    dt = t_p/100
    num_points = int(t_end/dt)
    initial_conditions = tf.constant([
        [r_1, 0.0, 0.0, l_1],
        [r_2, 0.0, 0.0, l_2]
    ], dtype=tf.float32)

    # verify dynamics

    generator = HamiltonianDataGenerator(nu=sym_mass_ratio, omega_ref=2)
    dydt = generator._dynamics(0, initial_conditions)
    
    omega_refs = [2]

    for omega_ref in omega_refs:
        # 1. Instantiate the generator
        generator = HamiltonianDataGenerator(nu=sym_mass_ratio, omega_ref=omega_ref)
        
        # 2. Call the generate_data method
        time_points, trajectory, h_data, derivatives = generator.generate_data(
            t_span=[t_start, t_end],
            t_points=num_points,
            x0=initial_conditions,
            noise_std=noise_level,
            return_amp_phase=True,
            derivatives=True
        )

        r_fin = trajectory[-1, 0, 0]
        r_init = r_2
        e_rel = np.abs((r_fin - r_init)/r_init)
        if e_rel < 1e-3:
            break
        
    # --- 3. Output and Verification ---
    print("--- DEBUGGED CLASS-BASED IMPLEMENTATION ---")
    print(f"Shape of time points: {time_points.shape}")
    print(f"Shape of output trajectory: {trajectory.shape}")
    print(f"Shape of final data [Amp, Phase]: {h_data.shape}")
    print(f"Optimal omega_ref: {omega_ref}")
    print("\n--- Final State (First Trajectory) ---")
    print(f"r: {trajectory[-1, 0, 0]:.2f}, phi: {trajectory[-1, 0, 1]:.2f}, p_r: {trajectory[-1, 0, 2]:.2f}, p_phi: {trajectory[-1, 0, 3]:.2f}")
    print(f"Amplitude: {h_data[-1, 0, 0]:.3f}, Unwrapped Phase: {h_data[-1, 0, 1]:.2f}")

    # --- 4. Plotting for Visual Verification ---
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')

        final_phase = h_data[:, 0, 1]
        phase_for_plot = final_phase.numpy()

        # plot the orbital dynamics. 
        # subplot of all 4 dynamical variables

        plt.figure(figsize=(12, 5))
        plt.suptitle("Orbital Dynamics (Debugged Class)")

        plt.subplot(2, 2, 1)
        plt.plot(time_points, trajectory[:, 0, 0], label='r', lw=1.5)
        plt.title("r")
        plt.xlabel("Time")
        plt.ylabel("r")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(time_points, trajectory[:, 0, 1], label='phi', lw=1.5)
        plt.title("phi")
        plt.xlabel("Time")
        plt.ylabel("phi")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(time_points, trajectory[:, 0, 2], label='p_r', lw=1.5)
        plt.title("p_r")
        plt.xlabel("Time")
        plt.ylabel("p_r")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(time_points, trajectory[:, 0, 3], label='p_phi', lw=1.5)
        plt.title("p_phi")
        plt.xlabel("Time")
        plt.ylabel("p_phi")
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # plot the derivatives

        plt.figure(figsize=(12, 5))
        plt.suptitle("Derivatives (Debugged Class)")
        plt.subplot(2, 2, 1)
        plt.plot(time_points, derivatives[:, 0, 0], label='dr/dt', lw=1.5)
        plt.title("dr/dt")
        plt.xlabel("Time")
        plt.ylabel("dr/dt")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(time_points, derivatives[:, 0, 1], label='dphi/dt', lw=1.5)
        plt.title("dphi/dt")
        plt.xlabel("Time")
        plt.ylabel("dphi/dt")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(time_points, derivatives[:, 0, 2], label='dp_r/dt', lw=1.5)
        plt.title("dp_r/dt")
        plt.xlabel("Time")
        plt.ylabel("dp_r/dt")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(time_points, derivatives[:, 0, 3], label='dp_phi/dt', lw=1.5)
        plt.title("dp_phi/dt")
        plt.xlabel("Time")
        plt.ylabel("dp_phi/dt")
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # plot the waveform

        plt.figure(figsize=(12, 5))
        plt.suptitle("Phase Unwrapping Verification (Debugged Class)")
        plt.subplot(1, 2, 1)
        plt.plot(time_points, phase_for_plot, label='Wrapped Phase', lw=1.5)
        plt.title("Wrapped Phase")
        plt.xlabel("Time")
        plt.ylabel("Phase (radians)")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(time_points, final_phase, label='Unwrapped Phase', color='darkorange', lw=1.5)
        plt.title("Unwrapped Phase")
        plt.xlabel("Time")
        plt.ylabel("Phase (radians)")
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.suptitle("Amplitude Verification (Debugged Class)")
        plt.subplot(1, 2, 1)
        plt.plot(time_points, h_data[:, 0, 0], label='Amplitude', lw=1.5)
        plt.title("Amplitude")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(time_points, h_data[:, 1, 0], label='Amplitude', lw=1.5)
        plt.title("Amplitude")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plot.")
