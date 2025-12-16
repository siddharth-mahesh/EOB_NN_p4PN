import jax
import jax.numpy as jnp
from jax.numpy import log
import jax.scipy.special as jss
from EOB_NN_p4PN.gamma import gamma as tgamma
jax.config.update("jax_enable_x64", True)
I = 1j

def _strain(self,point, nu, constants):
    """
    Calculate the EOB factorized resummed strain.

    Args:
        point (jnp.ndarray): point in the trajectory [r,phi,p_r,p_phi].
        nu (float): symmetric mass ratio
        constants (Dict): dictionary of EOB constants

    Returns:
        strain (complex): Complex GW strain.
    """
    _, phi, _, _ = point
    f_1 = constants["f_1"]
    f_2 = constants["f_2"]
    f_3 = constants["f_3"]
    f_3_l = constants["f_3_l"]
    delta_1_5 = constants["delta_1_5"]
    delta_2_5 = constants["delta_2_5"]
    delta_3 = constants["delta_3"]
    pi = constants["pi"]
    e_gamma = constants["e_gamma"]
    r0 = 2 / jnp.sqrt(jnp.e)
    # t and r_ISCO don't matter here
    Omega = self._eom(0, point, (nu, 0, constants))[1]
    H = nu * self._hamiltonian(point, nu, constants)
    tmp0 = jnp.pow(Omega, 2.0 / 3.0)
    tmp2 = 4 * I * H * Omega
    h22 = (
        (4.0 / 5.0)
        * jnp.sqrt(5)
        * nu
        * jnp.sqrt(pi)
        * tmp0
        * (1 + (1.0 / 2.0) * (((H) * (H)) - 1) / nu)
        * (
            jnp.pow(Omega, 4.0 / 3.0) * f_2
            + ((Omega) * (Omega)) * (f_3 + f_3_l * jnp.log(jnp.cbrt(Omega)))
            + f_1 * tmp0
            + 1
        )
        * tgamma(3 - tmp2)
        * jnp.exp(-2 * I * phi)
        * jnp.exp(
            I
            * (
                jnp.pow(Omega, 5.0 / 3.0) * delta_2_5
                + ((Omega) * (Omega)) * delta_3
                + Omega * delta_1_5
            )
        )
        * jnp.exp(2 * H * Omega * pi + tmp2 * jnp.log(4 * Omega * r0))
    )

    return h22