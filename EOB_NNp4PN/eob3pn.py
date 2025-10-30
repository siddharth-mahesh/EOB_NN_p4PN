"""
This file contains the EOB3PN class,
which implements the non-spinning 3PN Effective One Body model
with 3.5PN circular radiation-reaction.
"""

import jax
import jax.numpy as jnp
from jax.numpy import log

# set jax to 64 bit precision
jax.config.update("jax_enable_x64", True)
import diffrax
import optimistix
import equinox as eqx


class EOB:
    def __init__(self):
        """
        Initialize the EOB class.
        """

        # model identifiers
        self.conservative_order = 3
        self.radiative_order = 3.5

    def _pade_1_3(self, x, a_1, a_3, a_4):
        """
        Compute the Pade approximant P^1_3 for the Hamiltonian A potential.
        The Hamiltonian A potential is given by a polynomial of the form
        p(x) = 1 + a_1 x + a_3 x^3 + a_4 x^4

        Args:
            x (float): Input tensor, typically 1/r.
            a_1 (float): Coefficient a_1.
            a_3 (float): Coefficient a_3.
            a_4 (float): Coefficient a_4.

        Returns:
            float: Pade approximant P^1_3 evaluated at x.
        """
        tmp1 = a_1 * a_3
        tmp2 = 1.0 / (((a_1) * (a_1) * (a_1)) + a_3)
        tmp4 = -a_4 + tmp1
        pade_1_3 = (
            tmp2 * x * (((a_1) * (a_1) * (a_1) * (a_1)) - a_4 + 2 * tmp1) + 1
        ) / (
            -a_1 * tmp2 * tmp4 * ((x) * (x))
            + tmp2 * tmp4 * x
            + tmp2 * ((x) * (x) * (x)) * (-((a_1) * (a_1)) * a_4 - ((a_3) * (a_3)))
            + 1
        )
        return pade_1_3

    def _pade_0_3(self, x, d_2, d_3):
        """
        Compute the Pade approximant P^{0}_{3} for the polynomial
        p(x) = 1 + d_2 x^2 + d_3 x^3

        Args:
            x (float): Input tensor, typically 1/r.
            d_2 (float): Coefficient d_2.
            d_3 (float): Coefficient d_3.

        Returns:
            float: Pade approximant P^{0}_{3} evaluated at x.
        """
        return 1 / (1 - x * x * (d_3 * x + d_2))

    def _set_eob_constants_3PN(self, nu):
        """
        Calculate the dictionary of EOB constants.

        Args:
            nu (float): Symmetric mass ratio.

        Returns:
            dict: Dictionary of EOB constants.
        """
        # All constants are batched with nu so need to define unit_tensor to broadcast constants that are not nu-dependent
        e_gamma = 0.577215664901532860606512090082402431042
        pi = 3.14159265358979323846264338327950288419716939937510
        theta_hat = 1039/4620
        tmp0 = 2*nu
        tmp1 = ((pi)*(pi))
        tmp3 = 35*nu - 36
        tmp5 = ((nu)*(nu))
        tmp7 = ((nu)*(nu)*(nu))
        tmp9 = (nu + 3)*(-7956*nu + tmp3*(tmp0 + 6)*jnp.sqrt(-81*nu + 4*tmp5 + 144) + 2979*tmp5 - 140*tmp7 + 5184)/(tmp3*(7956*nu - 2979*tmp5 + 140*tmp7 - 5184))
        a_1 = -2
        a_3 = tmp0
        a_4 = nu*(94.0/3.0 - 41.0/32.0*tmp1)
        z_3 = nu*(8 - 6*nu)
        d_2 = -6*nu
        d_3 = tmp0*(3*nu - 26)
        v_meco_p2 = 2*jnp.sqrt(tmp9)
        v_pole_p2 = jnp.sqrt(((1.0/3.0)*nu + 1)/(3 - 35.0/12.0*nu))
        F_2 = -35.0/12.0*nu - 1247.0/336.0
        F_3 = 4*pi
        F_4 = (9271.0/504.0)*nu + (65.0/18.0)*tmp5 - 44711.0/9072.0
        F_5 = pi*(-583.0/24.0*nu - 8191.0/672.0)
        F_6 = -1712.0/105.0*e_gamma + nu*(-88.0/3.0*theta_hat + (41.0/48.0)*tmp1 - 2913613.0/272160.0) + (16.0/3.0)*tmp1 - 94403.0/3024.0*tmp5 - 775.0/324.0*tmp7 - 856.0/105.0*log(64*tmp9) + 6643739519.0/69854400.0
        F_6_l = -856.0/105.0
        F_7 = pi*((214745.0/1728.0)*nu + (193385.0/3024.0)*tmp5 - 16285.0/504.0)

        return {
            'e_gamma': e_gamma, 'pi': pi, 'theta_hat': theta_hat,
            'a_1': a_1, 'a_3': a_3, 'a_4': a_4, 'z_3': z_3,
            'd_2': d_2, 'd_3': d_3,
            'v_meco_p2': v_meco_p2, 'v_pole_p2': v_pole_p2,
            'F_2': F_2, 'F_3': F_3, 'F_4': F_4, 'F_5': F_5, 'F_6': F_6,
            'F_6_l': F_6_l, 'F_7': F_7
        }

    def _a_potential(self, r, constants):
        """
        Compute the Hamiltonian A potential.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Hamiltonian A potential
        """
        u = 1 / r
        a = self._pade_1_3(u, constants["a_1"], constants["a_3"], constants["a_4"])
        return a

    def _d_potential(self, r, constants):
        """
        Compute the Hamiltonian D potential.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Hamiltonian D potential
        """
        u = 1 / r
        d = self._pade_0_3(u, constants["d_2"], constants["d_3"])
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
        z_3 = constants['z_3']
        a = self._a_potential(r, constants)
        d = self._d_potential(r, constants)
        h_real = jnp.sqrt(2*nu*(jnp.sqrt(a*(((p_phi)*(p_phi))*((u)*(u)) + ((p_r)*(p_r))*(a/d + ((p_r)*(p_r))*((u)*(u))*z_3) + 1)) - 1) + 1)/nu
        return h_real


    def _flux(self, v, nu, constants):
        """
        Compute the circular gravitational flux at the 3.5 PN order

        Args:
            v (float): Orbital velocity
            nu (float): Compactness
            constants (dict): Dictionary of constants

        Returns:
            float: Gravitational flux
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
        M_LN2 = jnp.log(2.0)
        tmp0 = (1.0/(v_p))
        tmp2 = F_2*v_p
        tmp5 = (1.0/((v_p)*(v_p)))
        tmp21 = (1.0/((v_p)*(v_p)*(v_p)))
        tmp51 = (1.0/((v_p)*(v_p)*(v_p)*(v_p)))
        tmp111 = (1.0/((v_p)*(v_p)*(v_p)*(v_p)*(v_p)))
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
        tmp121 = -tmp32*tmp35 - tmp32*tmp49 - tmp33 - tmp34 - tmp38*tmp46 - tmp39 - tmp41*tmp42 - tmp42*tmp44 - tmp53 + tmp54 - tmp56 - tmp59 - tmp60
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
        tmp120 = F_6*tmp109*v_p
        tmp122 = -F_5*tmp109 - tmp101 - tmp103 - tmp106 - tmp108 - tmp109*tmp111 - tmp11 - tmp114 - tmp116 - tmp118 - tmp119 + tmp120 - tmp121 - tmp44*tmp78 - tmp46*tmp71 - tmp49*tmp64 - tmp61 - tmp64*tmp74 - tmp65 - tmp66 - tmp67 - tmp69 - tmp72 - tmp77*tmp78 - tmp83 - tmp86 - tmp89 - tmp91*tmp92 - tmp97 - tmp98 - tmp99
        tmp123 = (1.0/(tmp122))
        tmp124 = tmp123*tmp64
        tmp125 = tmp123*tmp30
        tmp126 = tmp123*tmp62
        tmp130 = tmp123*tmp71
        tmp131 = tmp123*tmp65
        tmp132 = tmp124*tmp13
        tmp134 = tmp124*tmp38
        tmp136 = tmp123*tmp13*tmp78
        tmp138 = 20*tmp130
        flux = -32.0/5.0*nu*jnp.pow(v, 7)*(2*F_6_l*jnp.pow(v, 6)*jnp.log(v/v_m) + 1)/((-tmp0*v + 1)*(tmp0*v/(tmp4*v/(tmp13*v/(tmp30*v/(tmp62*v/(tmp122*v/(v*(F_5*tmp109 + F_6*tmp123*tmp16*tmp32*tmp6*tmp64 - F_7*tmp130*tmp22*v_p - 24*tmp0*tmp124*tmp15*tmp42 - 18*tmp0*tmp132*tmp4 - tmp0*tmp138*tmp37 + tmp101 + tmp103 - tmp104*tmp124*tmp15 - tmp104*tmp124*tmp33*tmp4 + tmp106 + tmp108 + tmp109*tmp111 - tmp11 - 6*tmp111*tmp124*tmp38 + tmp114 + tmp116 + tmp118 + tmp119 - tmp120 - tmp121 - tmp123*tmp13*tmp55 - tmp123*tmp18 - 3*tmp123*((tmp30)*(tmp30)) - 6*tmp123*tmp31*tmp78 - 3*tmp123*tmp31 - tmp123*tmp33*tmp66 - tmp123*tmp35 - tmp123*tmp44*tmp72 - tmp123*tmp49*tmp78 - tmp123*tmp49 - tmp123*((tmp62)*(tmp62)) - tmp123*tmp72*(-5*tmp0 + 5*tmp2) - tmp123*tmp74 - 10*tmp124*tmp15*tmp33 - tmp124*((tmp30)*(tmp30)*(tmp30)) - tmp124*tmp31*tmp48 - tmp124*tmp31*(-tmp104 + 12*tmp2) - 10*tmp124*tmp37*tmp42 - 4*tmp124*tmp37 - 15*tmp124*tmp39*tmp5 - tmp124*tmp42*tmp46 - 18*tmp124*tmp42*tmp57 - tmp124*tmp46 - 12*tmp124*tmp57 - 4*tmp124*tmp70 - tmp125*tmp55 - tmp125*tmp66 - tmp125*tmp67 - tmp125*tmp95 - tmp125*(6*F_2*tmp6 - 6*tmp10 - 6*tmp2 - tmp48 - 6*tmp7) - tmp126*tmp44 - tmp126*tmp77 - tmp126*(2*F_2*tmp6 - 2*tmp10 - 2*tmp2 - tmp44 - 2*tmp7) - tmp126*(3*F_4*tmp16*tmp6*v_p + 3*tmp10 - tmp115*tmp16*tmp4 - 9*tmp16*tmp5 - 3*tmp17 - 3*tmp20 - 3*tmp23 - tmp41 + 3*tmp7 - 3*tmp8) - ((tmp13)*(tmp13)*(tmp13)*(tmp13))*tmp130 - 30*tmp130*tmp15*tmp5 - tmp130*tmp22/jnp.pow(v_p, 6) - 5*tmp130*tmp51 - 5*tmp130*tmp91 - tmp131*tmp44 - tmp131*tmp77 - tmp131*(4*F_2*tmp6 - 4*tmp10 - 4*tmp2 - tmp55 - 4*tmp7) - 12*tmp132*tmp15 - 6*tmp132*tmp5 - tmp134*((tmp4)*(tmp4)*(tmp4)*(tmp4)*(tmp4)) - 15*tmp134*tmp4*tmp51 - tmp134*tmp48*tmp91 - tmp136*tmp48 - tmp136*(-tmp115 + 9*tmp2) - tmp138*tmp17*tmp21 - tmp138*tmp21*tmp4 - tmp2 - tmp29 + tmp44*tmp78 + tmp46*tmp71 + tmp49*tmp64 + tmp64*tmp74 + tmp65 + tmp66 + tmp67 + tmp69 + tmp72 + tmp77*tmp78 + tmp83 + tmp86 + tmp89 + tmp91*tmp92 + tmp97 + tmp98 + tmp99) + 1) + 1) + 1) + 1) + 1) + 1) + 1))

        return flux

    
    def _eom(self,t,y,args):
        """
        The equations of motion for the EOB model.

        Args:
            t (float): Time.
            y (jnp.ndarray): Canonical variables [r, phi, p_rstar, p_phi].
            args (tuple): Additional parameters (nu, r_ISCO, constants).

        Returns:
            jnp.ndarray: Equations of motion.
        """
        nu , _ , constants = args
        num_coords = len(y)//2
        symplectic_map = jnp.block(
            [
                [jnp.zeros((num_coords,num_coords)),jnp.eye(num_coords)],
                [-jnp.eye(num_coords),jnp.zeros((num_coords,num_coords))]
            ]
        )
        d_h_real = jax.grad(self._hamiltonian, argnums=0)(y, nu, constants)
        omega = d_h_real[3] # omega = d_h_real/d_p_phi
        v = omega**(1./3.)
        flux = self._flux(v, nu, constants)
        ydot = symplectic_map @ d_h_real
        ydot = ydot.at[3].set(flux)
        return ydot

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
        a = self._a_potential(r, constants)
        d = self._d_potential(r, constants)
        tmp0 = ((p_phi)*(p_phi))*((u)*(u))
        tmp2 = 2*nu*(jnp.sqrt(a)*jnp.sqrt(tmp0 + 1) - 1)
        c_circ = jnp.pow(a, 3.0/2.0)/(d*jnp.sqrt(tmp0*tmp2 + tmp0 + tmp2 + 1))
        return c_circ

    def _j(self, r, constants):
        """
        Compute the circular orbit angular momentum.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Circular orbit angular momentum
        """
        r3 = r * r * r
        a = self._a_potential(r, constants)
        da_dr = jax.grad(self._a_potential, argnums=0)(r, constants)
        j = jnp.sqrt(r3 * da_dr / (2 * a - r * da_dr))
        return j

    def _w_circ(self, r, p_phi, nu, constants):
        """
        Compute the circular frequency.

        Args:
            r (float): Radial position.
            p_phi (float): Angular momentum.
            nu (float): Symmetric mass ratio.
            constants (dict): Dictionary of constants.

        Returns:
            float: Circular frequency.
        """
        u = 1 / (r + 1e-100)
        a = self._a_potential(r, constants)
        d = self._d_potential(r, constants)
        tmp0 = jnp.sqrt(a)
        tmp2 = ((p_phi)*(p_phi))*((u)*(u))
        tmp4 = 2*nu*(tmp0*jnp.sqrt(tmp2 + 1) - 1)
        w_circ = p_phi*tmp0*((u)*(u))/jnp.sqrt(tmp2*tmp4 + tmp2 + tmp4 + 1)
        return w_circ
    
    def _h_circ(self,r,nu,constants):
        """
        Compute the circular orbit Hamiltonian.

        Args:
            r (float): Radial position
            nu (float): Symmetric mass ratio
            constants (dict): Dictionary of constants

        Returns:
            float: Circular orbit Hamiltonian
        """
        j = self._j(r, constants)
        z_circ = jnp.array([r, 0.0, 0.0, j])
        h_circ = self._hamiltonian(z_circ, nu, constants)
        return h_circ

    def _pr_adiabatic(self, r, nu, constants):
        """
        Compute the radial momentum in the adiabatic limit.

        Args:
            r (float): Radial position
            constants (dict): Dictionary of constants.

        Returns:
            float: Radial momentum
        """
        j , dj_dr = jax.value_and_grad(self._j, argnums=0)(r, constants)
        c = self._c_potential(r, j, nu, constants)
        phidot = self._w_circ(r, j, nu, constants)
        v_w = jnp.pow(phidot, 1 / 3)
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
        j = self._j(r, constants)
        phidot = self._w_circ(r, j, nu, constants)
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
    
    def _isco_finder(self, nu, constants):
        """
        Find the ISCO radius for a given nu and constants.

        Args:
            nu (float): Symmetric mass ratio
            constants (dict): Dictionary of constants

        Returns:
            float: ISCO radius
        """
        r0 = 6.0
        params = (nu, constants)
        f = self._isco_condition(r0, params)
        for _ in range(100):
            if jnp.abs(f) < 1e-12:
                return r0
            f , df = jax.value_and_grad(self._isco_condition, argnums=0)(r0, params)
            r0 -= f / df
        raise ValueError(f"Newton's method did not converge, final error: {f}")

    def _circular_orbit_finder(self, r0, params, tol=1e-12, max_iter=100):
        """
        Finds the radius of a circular orbit using a Newton's method.
        This function solves for r given eta, omega_0, and constants.

        Args:
            r0 (float): Initial guess for the radius of the circular orbit.
            params (tuple): Parameters given by (eta, omega_0, constants).
            tol (float): The tolerance for the solution.
            max_iter (int): The maximum number of iterations to perform.

        Returns:
            float: Radius of the circular orbit
        """
        f = self._circular_orbit_condition(r0, params)
        for _ in range(max_iter):
            if jnp.abs(f) < tol:
                return r0
            f , df = jax.value_and_grad(self._circular_orbit_condition, argnums=0)(r0, params)
            r0 -= f / df
        raise ValueError(f"Newton's method did not converge, final error: {f}")

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
        r0 = self._circular_orbit_finder(omega_0**(-2/3), (nu, omega_0, constants))
        pr0 = self._pr_adiabatic(r0, nu, constants)
        pphi0 = self._j(r0, constants)
        return jnp.array([r0, 0.0, pr0, pphi0])
    
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
        _ , r_ISCO , _ = args
        r , _ , _ , _ = y
        return r - r_ISCO

    def _dynamics(self,y0, nu, constants, dt = 0.1):
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
        r_ISCO = self._isco_finder(nu, constants)
        print(y0)
        print(r_ISCO)
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
            event=diffrax.Event(self._event_fn, optimistix.Newton(1e-5,1e-5,optimistix.rms_norm)),
            saveat=diffrax.SaveAt(t0=True,t1=True,dense=True)
        )
        times = jnp.linspace(0, sol.ts[-1], 10000)
        trajectory = jax.vmap(sol.evaluate, in_axes=0)(times)
        return times, trajectory
    def __call__(self, x):
        nu = x[0]
        constants = self._set_eob_constants_3PN(nu)
        ics = self._initial_conditions(x)
        return self._dynamics(ics, nu, constants)
        

if __name__ == "__main__":
    eob3pn = EOB()
    times ,  trajectory = eob3pn(jnp.array([0.25, 0.01]))
    import matplotlib.pyplot as plt
    plt.plot(trajectory[:,0]*jnp.cos(trajectory[:,1]), trajectory[:,0]*jnp.sin(trajectory[:,1]))
    plt.show()