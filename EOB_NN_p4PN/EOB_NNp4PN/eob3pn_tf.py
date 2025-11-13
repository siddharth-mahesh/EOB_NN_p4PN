"""
    This file contains the EOB3PN class,
    which implements the non-spinning 3PN Effective One Body model
    with 3.5PN circular radiation-reaction.
"""

# Core libraries
import tensorflow as tf 
import numpy as np

class EOB(tf.Module):
    def __init__(self, name=None):
        """
        Initialize the EOB class.

        Args:
            name (str): Name of the module.
        """

        super().__init__(name=name)
        # Some settings to make this class extensible to 4 PN
        # These will be extended with higher orders in the future
        self.conservative_order = 3
        self.radiative_order = 3.5
        self.pade = self._pade_1_3

    @tf.function
    def _pade_1_3(self,x,a_1, a_3, a_4):
        """
        Compute the Pade approximant P^1_3 for the Hamiltonian A potential.
        The Hamiltonian A potential is given by a polynomial of the form
        p(x) = 1 + a_1 x + a_3 x^3 + a_4 x^4

        Args:
            x (tf.Tensor): Input tensor, typically 1/r (None,).
            a_1 (tf.Tensor): Coefficient a_1 (None,).
            a_3 (tf.Tensor): Coefficient a_3 (None,).
            a_4 (tf.Tensor): Coefficient a_4 (None,).

        Returns:
            tf.Tensor: Pade approximant P^1_3 evaluated at x (None,).
        """
        tmp1 = a_1*a_3
        tmp2 = (1.0/(((a_1)*(a_1)*(a_1)) + a_3))
        tmp4 = -a_4 + tmp1
        pade_1_3 = (tmp2*x*(((a_1)*(a_1)*(a_1)*(a_1)) - a_4 + 2*tmp1) + 1)/(-a_1*tmp2*tmp4*((x)*(x)) + tmp2*tmp4*x + tmp2*((x)*(x)*(x))*(-((a_1)*(a_1))*a_4 - ((a_3)*(a_3))) + 1)
        return pade_1_3
    
    @tf.function
    def _pade_0_3(self, x, d_2, d_3):
        """
        Compute the Pade approximant P^{0}_{3} for the polynomial
        p(x) = 1 + d_2 x^2 + d_3 x^3

        Args:
            x (tf.Tensor): Input tensor, typically 1/r (None,).
            d_2 (tf.Tensor): Coefficient d_2 (None,).
            d_3 (tf.Tensor): Coefficient d_3 (None,).

        Returns:
            tf.Tensor: Pade approximant P^{0}_{3} evaluated at x (None,).
        """
        return 1 / (1 - x * x * (d_3 * x + d_2) + 1e-100)

    @tf.function
    def _set_eob_constants_3PN(self, nu):
        """
        Calculate the dictionary of EOB constants.

        Args:
            nu (tf.Tensor): Symmetric mass ratio (None,).

        Returns:
            dict: Dictionary of EOB constants.
        """
        # All constants are batched with nu so need to define unit_tensor to broadcast constants that are not nu-dependent
        unit_tensor = tf.ones_like(nu)
        e_gamma = tf.constant(0.577215664901532860606512090082402431042, dtype=tf.float64)
        pi = tf.constant(3.14159265358979323846264338327950288419716939937510, dtype=tf.float64)
        theta_hat = tf.constant(1039/4620, dtype=tf.float64)
        tmp0 = 2*nu
        tmp1 = ((pi)*(pi))
        tmp3 = 35*nu - 36
        tmp5 = ((nu)*(nu))
        tmp7 = ((nu)*(nu)*(nu))
        a_1 = -2*unit_tensor
        a_3 = tmp0
        a_4 = nu*(94.0/3.0 - 41.0/32.0*tmp1)
        z_3 = nu*(8 - 6*nu)
        d_2 = -6*nu
        d_3 = tmp0*(3*nu - 26)
        v_meco_p2 = 2*tf.math.sqrt((nu + 3)*(-7956*nu + tmp3*(tmp0 + 6)*tf.math.sqrt(-81*nu + 4*tmp5 + 144) + 2979*tmp5 - 140*tmp7 + 5184)/(tmp3*(7956*nu - 2979*tmp5 + 140*tmp7 - 5184)))
        v_pole_p2 = tf.math.sqrt(((1.0/3.0)*nu + 1)/(3 - 35.0/12.0*nu))
        F_2 = -35.0/12.0*nu - 1247.0/336.0
        F_3 = 4*pi*unit_tensor
        F_4 = (9271.0/504.0)*nu + (65.0/18.0)*tmp5 - 44711.0/9072.0
        F_5 = pi*(-583.0/24.0*nu - 8191.0/672.0)
        F_6 = -1712.0/105.0*e_gamma + nu*(-88.0/3.0*theta_hat + (41.0/48.0)*tmp1 - 2913613.0/272160.0) + (16.0/3.0)*tmp1 - 94403.0/3024.0*tmp5 - 775.0/324.0*tmp7 + 6643739519.0/69854400.0
        F_6_l = -856.0/105.0*unit_tensor
        F_7 = pi*((214745.0/1728.0)*nu + (193385.0/3024.0)*tmp5 - 16285.0/504.0)
        # Cast all constants to tf.float64
        return {
            'e_gamma': e_gamma, 'pi': pi, 'theta_hat': theta_hat,
            'a_1': tf.cast(a_1, dtype=tf.float64), 'a_3': tf.cast(a_3, dtype=tf.float64), 'a_4': tf.cast(a_4, dtype=tf.float64), 'z_3': tf.cast(z_3, dtype=tf.float64),
            'd_2': tf.cast(d_2, dtype=tf.float64), 'd_3': tf.cast(d_3, dtype=tf.float64),
            'v_meco_p2': tf.cast(v_meco_p2, dtype=tf.float64), 'v_pole_p2': tf.cast(v_pole_p2, dtype=tf.float64),
            'F_2': tf.cast(F_2, dtype=tf.float64), 'F_3': tf.cast(F_3, dtype=tf.float64), 'F_4': tf.cast(F_4, dtype=tf.float64), 'F_5': tf.cast(F_5, dtype=tf.float64), 'F_6': tf.cast(F_6, dtype=tf.float64),
            'F_6_l': tf.cast(F_6_l, dtype=tf.float64), 'F_7': tf.cast(F_7, dtype=tf.float64)
        }
    @tf.function
    def _a_potential(self, r, constants):
        """
        Compute the Hamiltonian A potential.

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Hamiltonian A potential (batch_size,)
        """
        u = 1 / r
        a = self.pade(u, constants['a_1'], constants['a_3'], constants['a_4'])
        return a
    
    @tf.function
    def _d_potential(self, r, constants):
        """
        Compute the Hamiltonian D potential.

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Hamiltonian D potential (batch_size,)
        """
        u = 1 / r
        d = self._pade_0_3(u, constants['d_2'], constants['d_3'])
        return d

    @tf.function
    def _hamiltonian(self, x, nu, constants):
        """
        Compute the Hamiltonian.

        Args:
            x (tf.Tensor): Canonical variables [r, phi, p_r, p_phi] (batch_size,4).
            nu (tf.Tensor): Symmetric mass ratio (batch_size,).
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Hamiltonian evaluated at x (batch_size,1).
        """
        r , _ , p_r , p_phi = tf.split(x,4,axis=-1)
        u = 1 / r
        z_3 = tf.zeros_like(nu)
        a = self._a_potential(r, constants)
        d = self._d_potential(r, constants)
        h_real = tf.math.sqrt(2*nu*(tf.math.sqrt(a*(((p_phi)*(p_phi))*((u)*(u)) + ((p_r)*(p_r))*(a/d + ((p_r)*(p_r))*((u)*(u))*z_3) + 1)) - 1) + 1)/nu
        return tf.cast(h_real, dtype=tf.float64)

    @tf.function
    def _flux(self, v, nu, constants):
        """
        Compute the circular gravitational flux at the 3.5 PN order

        Args:
            v (tf.Tensor): Orbital velocity (None,)
            nu (tf.Tensor): Compactness (None,)
            constants (dict): Dictionary of constants

        Returns:
            tf.Tensor: Gravitational flux (None,1)
        """
        F_2 = constants['F_2']
        F_3 = constants['F_3']
        F_4 = constants['F_4']
        F_5 = constants['F_5']
        F_6 = constants['F_6']
        F_6_l = constants['F_6_l']
        F_7 = constants['F_7']
        v_p = constants['v_pole_p2']
        v_m = constants['v_meco_p2']
        M_LN2 = tf.cast(tf.math.log(2.0), tf.float64)
        tmp0 = (1.0/(v_p))
        tmp2 = F_2*v_p
        tmp5 = (1.0/((v_p)*(v_p)))
        tmp21 = (1.0/((v_p)*(v_p)*(v_p)))
        tmp51 = (1.0/((v_p)*(v_p)*(v_p)*(v_p)))
        tmp111 = (1.0/((v_p)*(v_p)*(v_p)*(v_p)*(v_p)))
        tmp122 = tf.math.log(((v_m)*(v_m)))
        tmp124 = M_LN2
        tmp18 = 3*tmp5
        tmp24 = 3*tmp0
        tmp44 = 2*tmp0
        tmp46 = 4*tmp21
        tmp48 = 6*tmp0
        tmp55 = 4*tmp0
        tmp104 = 12*tmp0
        tmp115 = 9*tmp0
        tmp4 = -tmp0 + tmp2
        tmp41 = 3*tmp2 - tmp24
        tmp77 = 2*tmp2 - tmp44
        tmp95 = 4*tmp2 - tmp55
        tmp6 = (1.0/(tmp4))
        tmp15 = ((tmp4)*(tmp4))
        tmp37 = ((tmp4)*(tmp4)*(tmp4))
        tmp49 = tmp4*tmp48
        tmp57 = tmp4*tmp5
        tmp91 = ((tmp4)*(tmp4)*(tmp4)*(tmp4))
        tmp7 = tmp5*tmp6
        tmp8 = F_2*tmp6
        tmp35 = 3*tmp15
        tmp10 = F_3*tmp6*v_p
        tmp11 = tmp0 + tmp10 + tmp7 - tmp8
        tmp14 = -tmp10 - tmp7 + tmp8
        tmp13 = -tmp11 - tmp2
        tmp16 = (1.0/(tmp13))
        tmp31 = ((tmp13)*(tmp13))
        tmp70 = ((tmp13)*(tmp13)*(tmp13))
        tmp74 = tmp13*(6*tmp2 - tmp48)
        tmp17 = tmp15*tmp16
        tmp20 = F_3*tmp16*tmp6
        tmp22 = tmp16*tmp6
        tmp23 = tmp21*tmp22
        tmp26 = tmp16*tmp24*tmp4
        tmp28 = F_4*tmp22*v_p
        tmp29 = tmp16*tmp18 + tmp17 + tmp20 + tmp23 + tmp26 - tmp28
        tmp61 = -tmp16*tmp18 - tmp17 - tmp2 - tmp20 - tmp23 - tmp26 + tmp28
        tmp30 = -tmp14 - tmp29 - tmp4
        tmp32 = (1.0/(tmp30))
        tmp33 = tmp31*tmp32
        tmp34 = tmp18*tmp32
        tmp38 = tmp16*tmp32
        tmp42 = tmp13*tmp32
        tmp54 = F_4*tmp22*tmp32
        tmp56 = tmp17*tmp32*tmp55
        tmp39 = tmp37*tmp38
        tmp53 = tmp22*tmp32*tmp51
        tmp60 = F_5*tmp22*tmp32*v_p
        tmp59 = 6*tmp38*tmp57
        tmp62 = tmp0 - tmp14 - tmp32*tmp35 - tmp32*tmp49 - tmp33 - tmp34 - tmp38*tmp46 - tmp39 - tmp41*tmp42 - tmp42*tmp44 - tmp53 + tmp54 - tmp56 - tmp59 - tmp60 - tmp61
        tmp126 = -tmp32*tmp35 - tmp32*tmp49 - tmp33 - tmp34 - tmp38*tmp46 - tmp39 - tmp41*tmp42 - tmp42*tmp44 - tmp53 + tmp54 - tmp56 - tmp59 - tmp60
        tmp64 = (1.0/(tmp62))
        tmp65 = ((tmp30)*(tmp30))*tmp64
        tmp66 = tmp18*tmp64
        tmp67 = tmp35*tmp64
        tmp69 = 3*tmp31*tmp64
        tmp71 = tmp32*tmp64
        tmp78 = tmp30*tmp64
        tmp86 = tmp13*tmp55*tmp64
        tmp92 = tmp38*tmp64
        tmp99 = tmp13*tmp34*tmp64
        tmp114 = 5*tmp0*tmp39*tmp64
        tmp72 = tmp70*tmp71
        tmp83 = tmp78*(-3*tmp10 - 3*tmp2 - tmp24 - 3*tmp7 + 3*tmp8)
        tmp89 = 4*tmp37*tmp71
        tmp97 = tmp33*tmp64*tmp95
        tmp98 = tmp33*tmp44*tmp64
        tmp101 = 5*tmp51*tmp92
        tmp103 = 6*tmp15*tmp42*tmp64
        tmp106 = tmp104*tmp15*tmp71
        tmp108 = 12*tmp57*tmp71
        tmp109 = tmp22*tmp71
        tmp116 = tmp115*tmp4*tmp42*tmp64
        tmp118 = 10*tmp21*tmp4*tmp92
        tmp119 = 10*tmp17*tmp5*tmp71
        tmp120 = tmp109*v_p
        tmp123 = F_6_l*tmp120*tmp122
        tmp125 = 4*F_6_l*tmp120*tmp124
        tmp127 = -F_5*tmp109 + F_6*tmp120 - tmp101 - tmp103 - tmp106 - tmp108 - tmp109*tmp111 - tmp11 - tmp114 - tmp116 - tmp118 - tmp119 + tmp123 + tmp125 - tmp126 - tmp44*tmp78 - tmp46*tmp71 - tmp49*tmp64 - tmp61 - tmp64*tmp74 - tmp65 - tmp66 - tmp67 - tmp69 - tmp72 - tmp77*tmp78 - tmp83 - tmp86 - tmp89 - tmp91*tmp92 - tmp97 - tmp98 - tmp99
        tmp128 = (1.0/(tmp127))
        tmp129 = tmp128*tmp64
        tmp130 = tmp128*tmp30
        tmp131 = tmp128*tmp62
        tmp135 = tmp128*tmp71
        tmp136 = tmp128*tmp65
        tmp137 = tmp129*tmp13
        tmp139 = tmp129*tmp38
        tmp141 = tmp128*tmp13*tmp78
        tmp143 = 20*tmp135
        flux = -32.0/5.0*nu*tf.math.pow(v, 7)*(2*F_6_l*tf.math.log(v/v_m) + 1)/((-tmp0*v + 1)*(tmp0*v/(tmp4*v/(tmp13*v/(tmp30*v/(tmp62*v/(tmp127*v/(v*(F_5*tmp109 - F_6*tmp120 + F_6*tmp128*tmp16*tmp32*tmp6*tmp64 + F_6_l*tmp122*tmp128*tmp16*tmp32*tmp6*tmp64 + 4*F_6_l*tmp124*tmp128*tmp16*tmp32*tmp6*tmp64 - F_7*tmp135*tmp22*v_p - 24*tmp0*tmp129*tmp15*tmp42 - 18*tmp0*tmp137*tmp4 - tmp0*tmp143*tmp37 + tmp101 + tmp103 - tmp104*tmp129*tmp15 - tmp104*tmp129*tmp33*tmp4 + tmp106 + tmp108 + tmp109*tmp111 - tmp11 - 6*tmp111*tmp129*tmp38 + tmp114 + tmp116 + tmp118 + tmp119 - tmp123 - tmp125 - tmp126 - tmp128*tmp13*tmp55 - tmp128*tmp18 - 3*tmp128*((tmp30)*(tmp30)) - 6*tmp128*tmp31*tmp78 - 3*tmp128*tmp31 - tmp128*tmp33*tmp66 - tmp128*tmp35 - tmp128*tmp44*tmp72 - tmp128*tmp49*tmp78 - tmp128*tmp49 - tmp128*((tmp62)*(tmp62)) - tmp128*tmp72*(-5*tmp0 + 5*tmp2) - tmp128*tmp74 - 10*tmp129*tmp15*tmp33 - tmp129*((tmp30)*(tmp30)*(tmp30)) - tmp129*tmp31*tmp48 - tmp129*tmp31*(-tmp104 + 12*tmp2) - 10*tmp129*tmp37*tmp42 - 4*tmp129*tmp37 - 15*tmp129*tmp39*tmp5 - tmp129*tmp42*tmp46 - 18*tmp129*tmp42*tmp57 - tmp129*tmp46 - 12*tmp129*tmp57 - 4*tmp129*tmp70 - ((tmp13)*(tmp13)*(tmp13)*(tmp13))*tmp135 - tmp130*tmp55 - tmp130*tmp66 - tmp130*tmp67 - tmp130*tmp95 - tmp130*(6*F_2*tmp6 - 6*tmp10 - 6*tmp2 - tmp48 - 6*tmp7) - tmp131*tmp44 - tmp131*tmp77 - tmp131*(2*F_2*tmp6 - 2*tmp10 - 2*tmp2 - tmp44 - 2*tmp7) - tmp131*(3*F_4*tmp16*tmp6*v_p + 3*tmp10 - tmp115*tmp16*tmp4 - 9*tmp16*tmp5 - 3*tmp17 - 3*tmp20 - 3*tmp23 - tmp41 + 3*tmp7 - 3*tmp8) - 30*tmp135*tmp15*tmp5 - tmp135*tmp22/tf.math.pow(v_p, 6) - 5*tmp135*tmp51 - 5*tmp135*tmp91 - tmp136*tmp44 - tmp136*tmp77 - tmp136*(4*F_2*tmp6 - 4*tmp10 - 4*tmp2 - tmp55 - 4*tmp7) - 12*tmp137*tmp15 - 6*tmp137*tmp5 - tmp139*((tmp4)*(tmp4)*(tmp4)*(tmp4)*(tmp4)) - 15*tmp139*tmp4*tmp51 - tmp139*tmp48*tmp91 - tmp141*tmp48 - tmp141*(-tmp115 + 9*tmp2) - tmp143*tmp17*tmp21 - tmp143*tmp21*tmp4 - tmp2 - tmp29 + tmp44*tmp78 + tmp46*tmp71 + tmp49*tmp64 + tmp64*tmp74 + tmp65 + tmp66 + tmp67 + tmp69 + tmp72 + tmp77*tmp78 + tmp83 + tmp86 + tmp89 + tmp91*tmp92 + tmp97 + tmp98 + tmp99) + 1) + 1) + 1) + 1) + 1) + 1) + 1))

        return tf.expand_dims(tf.cast(flux, dtype=tf.float64), axis=-1)

    @tf.function
    def _strain(self, ys, dydts, nu):
        """
        Calculate the complex strain from the state and its derivatives.

        Args:
            ys (tf.Tensor): The state vector [r, phi, p_r, p_phi] of shape (num_timesteps, batch_size, 4).
            dydts (tf.Tensor): The time derivatives [dq_dt, dp_dt] of shape (num_timesteps, batch_size, 4).
            nu (tf.Tensor): Symmetric mass ratio (batch_size,).

        Returns:
            tf.Tensor: The complex strain h22 of shape (num_timesteps, batch_size).
        """
        # Split the state vector into position and momentum
        q, _ = tf.split(ys, 2, axis=-1)
        # Split the state derivatives into positional and momentum derivatives
        dqdt, _ = tf.split(dydts, 2, axis=-1)
        
        # Extract r and phi from the position vector
        r = q[..., 0] # (num_timesteps, batch_size)
        phi = q[..., 1] # (num_timesteps, batch_size)

        # Extract rdot and phidot from the positional derivatives
        rdot = dqdt[..., 0] # (num_timesteps, batch_size)
        phidot = dqdt[..., 1] # (num_timesteps, batch_size)
        
        # DEBUG 29: Correctly broadcast nu for element-wise operations.
        # nu is (batch_size,), need to expand dims to (1, batch_size) to broadcast with (num_timesteps, batch_size)
        nu_broadcasted = tf.expand_dims(nu, axis=0) 

        # DEBUG 30: Removed incorrect multiplication of r, phi, rdot, phidot by nu.
        # This was physically incorrect for strain calculation.

        # DEBUG 31: Added epsilon for numerical stability in division.
        real_part = 1 / (r + 1e-100) + r**2 * phidot**2 - rdot**2 
        imag_part = 2 * nu_broadcasted * rdot * phidot # nu_broadcasted will broadcast correctly
        H_22 = tf.complex(real_part, imag_part)
        
        # Calculate the complex strain h22
        # DEBUG 32: Completed the h22 calculation.
        h22 = 4 * tf.complex(nu_broadcasted, tf.zeros_like(nu_broadcasted,dtype=tf.float64)) * H_22 * tf.exp(tf.complex(tf.zeros_like(phi,dtype=tf.float64), -2.0 * phi))
        return h22

    @tf.function
    def _c_potential(self, r, p_phi, nu, constants):
        """
        Compute the Hamiltonian C potential.
        The C potential is given by:
            C = lim_{p_r -> 0} (1/p_r * dH/dp_r)

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            p_phi (tf.Tensor): Angular momentum (batch_size,)
            nu (tf.Tensor): Symmetric mass ratio (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Hamiltonian C potential (batch_size,)
        """
        u = 1 / r
        a = self._a_potential(r, constants)
        d = self._d_potential(r, constants)
        tmp0 = ((p_phi)*(p_phi))*((u)*(u))
        tmp2 = 2*nu*(tf.math.sqrt(a)*tf.math.sqrt(tmp0 + 1) - 1)
        c_circ = tf.math.pow(a, 3.0/2.0)/(d*tf.math.sqrt(tmp0*tmp2 + tmp0 + tmp2 + 1))
        return c_circ
    
    @tf.function
    def _j(self,r,constants):
        """
        Compute the circular orbit angular momentum.

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Circular orbit angular momentum (batch_size,)
        """
        r3 = r * r * r
        with tf.GradientTape() as tape:
            tape.watch(r)
            a = self._a_potential(r, constants)
        da_dr = tape.gradient(a, r)
        j = tf.math.sqrt(r3 * da_dr/(2 * a - r * da_dr))
        return j

    @tf.function
    def _dj_dr(self, r, constants):
        """
        Compute the derivative of the circular orbit angular momentum with respect to r.

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Derivative of circular orbit angular momentum with respect to r (batch_size,)
        """
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(r)
            r3 = r * r * r
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(r)
                a = self._a_potential(r, constants)
            da_dr = inner_tape.gradient(a, r)
            j = tf.math.sqrt(r3 * da_dr/(2 * a - r * da_dr))
        dj_dr = outer_tape.gradient(j, r)
        del inner_tape
        del outer_tape
        return dj_dr
    
    @tf.function
    def _w_circ(self, r, p_phi, nu, constants):
        """
        Compute the circular frequency.

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            p_phi (tf.Tensor): Angular momentum (batch_size,)
            nu (tf.Tensor): Symmetric mass ratio (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Circular frequency (batch_size,)
        """
        u = 1 / (r + 1e-100)
        a = self._a_potential(r, constants)
        d = self._d_potential(r, constants)
        tmp0 = tf.math.sqrt(a)
        tmp2 = ((p_phi)*(p_phi))*((u)*(u))
        tmp4 = 2*nu*(tmp0*tf.math.sqrt(tmp2 + 1) - 1)
        w_circ = p_phi*tmp0*((u)*(u))/tf.math.sqrt(tmp2*tmp4 + tmp2 + tmp4 + 1)
        return w_circ

    @tf.function
    def _pr_adiabatic(self,r,nu,constants):
        """
        Compute the radial momentum in the adiabatic limit.

        Args:
            r (tf.Tensor): Radial position (batch_size,)
            constants (dict): Dictionary of constants.

        Returns:
            tf.Tensor: Radial momentum (batch_size,)
        """
        j = self._j(r,constants)
        dj_dr = self._dj_dr(r,constants)
        c = self._c_potential(r,j,nu,constants)
        phidot = self._w_circ(r,j,nu,constants)
        v_w = tf.math.pow(phidot, 1/3)
        flux = tf.squeeze(self._flux(v_w,nu,constants), axis=-1)
        pr = flux/(c * dj_dr)
        return pr

    @tf.function
    def circular_orbit_condition(self, r, params):
        """
        Solve for the circular orbit condition for given nu and omega_0.
        
        Args:
            r (tf.Tensor): Radial position (batch_size,)
            params (tuple): Parameters given by (nu, omega_0, constants)

        Returns:
            tf.Tensor: Circular orbit condition (batch_size,)
        """
        nu , omega_0 , constants = params
        j = self._j(r,constants)
        phidot = self._w_circ(r,j,nu,constants)
        return phidot - omega_0
    
    @tf.function
    def solve_batch_powell(self, xi, x0, tol=1e-6, max_iter=100):
        """
        Finds the roots of a batch of functions using a modified Powell's hybrid method.

        This function solves for x in f(x; xi) = 0 for a batch of parameters xi.

        Args:
            xi (tuple): Parameters given by (nu, omega_0, constants).
            x0 (tf.Tensor): Initial guess for the roots (batch_size,).
            tol (float): The tolerance for the solution. The algorithm stops when the absolute value of f(x; xi) is less than tol.
            max_iter (int): The maximum number of iterations to perform.

        Returns:
            A tensor of shape (batch_size,) containing the roots x for each set of
            parameters xi.
        """
        x = x0
        i = tf.constant(0)
        # The condition for the while loop
        # The condition for the while loop, executed within the graph.
        def cond(i, x):
            y = self.circular_orbit_condition(x,xi)
            error = tf.abs(y)
            return tf.logical_and(
                tf.less(i, max_iter),
                tf.reduce_any(tf.greater(error, tol))
            )
        # The body of the while loop
        def body(i, x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = self.circular_orbit_condition(x,xi)
            # Calculate the Jacobian of f with respect to x
            jacobian = tape.gradient(y, x)
            # Avoid division by zero
            #safe_jacobian = tf.where(tf.equal(jacobian, 0), tf.ones_like(jacobian), jacobian)
            # Update x using the Newton-Raphson method (a simplified Powell step)
            x_new = x - y / jacobian
            return tf.add(i, 1), x_new
        # Run the while loop
        _, x_final = tf.while_loop(cond, body, [i, x])
        return x_final
    
    @tf.function
    def _dynamics(self, x, nu, constants):
        """
        Compute the equations of motion.
        The equations of motion are given by:
            dr/dt = dH / dpr
            dphi/dt = dH / dpphi
            dpr/dt = -dH / dr
            dpphi/dt = F

        Args:
            x (tf.Tensor): State vector [r, phi, p_r, p_phi] of shape (batch_size, 4)
            nu (tf.Tensor): Symmetric mass ratio (batch_size,)
            constants (dict): Dictionary of constants

        Returns:
            tf.Tensor: Equations of motion (batch_size, 4)
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            real_hamiltonian = self._hamiltonian(x, nu, constants)
        dH = tape.gradient(real_hamiltonian, x)
        
        # Return an error in cases where gradient might be None (e.g., non-differentiable ops).
        if dH is None:
            raise ValueError("Gradient of Hamiltonian is None!")
        dHdr , _ , dHdp_r , dHdp_phi = tf.split(dH,4,axis=-1)
        v_w = tf.squeeze(tf.math.pow(dHdp_phi, 1/3), axis=-1) 
        flux = self._flux(v_w, nu, constants)
        xdot = tf.concat([dHdp_r, dHdp_phi, -dHdr, flux],axis=-1)
        del tape
        return xdot
    
    @tf.function 
    def _initial_conditions(self,x_0_batch): 
        """ 
        Main routine to compute batched adiabatic initial conditions. 

        Args: 
            x_0_batch (tf.Tensor): Batch of symmetric mass ratio and frequency (batch_size,2) 

        Returns: 
            tuple: (r_0, pr_0, pphi_0) as tf.Tensor (batch_size,) 
        """ 
        nu , omega_0 = tf.split(x_0_batch,2,axis=-1) 
        nu = tf.squeeze(nu, axis=-1)
        omega_0 = tf.squeeze(omega_0, axis=-1) 
        r_0_initial_val = tf.math.pow(omega_0, -2/3)
        constants = self._set_eob_constants_3PN(nu)
        xi = (nu,omega_0,constants)
        r_0 = self.solve_batch_powell(xi,r_0_initial_val) 
        pphi_0 = self._j(r_0,constants)
        pr_0 = self._pr_adiabatic(r_0,nu,constants) 
        return r_0, pr_0, pphi_0
    
    @tf.function
    def __call__(self, x0_batch,ts,dt_initial,atol,rtol):
        """
        Main entry point for the EOB3PN model.
        
        Args:
            x0_batch (tf.Tensor): Batch of symmetric mass ratio and frequency (batch_size, 2).
            ts (tf.Tensor): Time points for integration (num_timesteps,).
            
        Returns:
            tuple: (trajectory, derivatives, strain)
        """
        nu, omega_0 = tf.split(x0_batch, 2, axis=-1)
        nu = tf.squeeze(nu, axis=-1) 
        omega_0 = tf.squeeze(omega_0, axis=-1) 
        constants = self._set_eob_constants_3PN(nu)
        # Compute initial conditions
        r0, pr0, pphi0 = self._initial_conditions(x0_batch)
        phi0 = tf.zeros_like(r0, dtype=tf.float64) 
        # Set initial state vector for integration (batch_size, 4)
        y0 = tf.concat([
            tf.expand_dims(r0, axis=-1),
            tf.expand_dims(phi0, axis=-1),
            tf.expand_dims(pr0, axis=-1),
            tf.expand_dims(pphi0, axis=-1)
        ], axis=-1)

        constants = self._set_eob_constants_3PN(nu)
        # Integrate the trajectory
        trajectory, derivatives, times = self._integrator_adaptive_euler(y0, nu,constants,ts[0],ts[-1],dt_initial,atol,rtol)
        
        # Calculate strain
        strain = self._strain(trajectory, derivatives, nu) 
        
        return trajectory, derivatives, times, strain

if __name__ == "__main__":
    eob3pn = EOB()
    batch_size = 3
    eta = tf.random.uniform((batch_size,1),minval=0.25,maxval=0.25,dtype=tf.float64)
    omega_0 = tf.random.uniform((batch_size,1),minval=0.01,maxval=0.012,dtype=tf.float64)
    dt = 2.0 * np.pi / omega_0 / 5.0
    dt_initial = tf.math.reduce_min(dt)
    ts = (0,1000)
    atol = 1e-6
    rtol = 1e-6
    x0 = tf.concat([eta,omega_0],axis=-1)
    trajectory, derivatives, times, strain = eob3pn(x0,ts,dt_initial,atol,rtol)
    import matplotlib.pyplot as plt
    t_to_plot = times.numpy()
    dt = np.diff(t_to_plot)
    #r_to_plot = trajectory[:,0,0].numpy()
    #phi_to_plot = trajectory[:,0,1].numpy()
    #x = r_to_plot * tf.math.cos(phi_to_plot)
    #y = r_to_plot * tf.math.sin(phi_to_plot)
    #plt.plot(x,y)
    plt.scatter(range(len(dt)),dt)
    plt.show()