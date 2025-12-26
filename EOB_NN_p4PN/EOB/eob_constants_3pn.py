import jax
import jax.numpy as jnp
from jax.numpy import log
jax.config.update("jax_enable_x64", True)
def set_eob_constants_3PN(nu):
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
    theta_hat = 1039 / 4620
    M_LN2 = jnp.log(2)
    tmp0 = 2 * nu
    tmp1 = (pi) * (pi)
    tmp3 = 35 * nu - 36
    tmp5 = (nu) * (nu)
    tmp7 = (nu) * (nu) * (nu)
    tmp9 = (
        (nu + 3)
        * (
            -7956 * nu
            + tmp3 * (tmp0 + 6) * jnp.sqrt(-81 * nu + 4 * tmp5 + 144)
            + 2979 * tmp5
            - 140 * tmp7
            + 5184
        )
        / (tmp3 * (7956 * nu - 2979 * tmp5 + 140 * tmp7 - 5184))
    )
    a_1 = -2
    a_3 = tmp0
    a_4 = nu * (94.0 / 3.0 - 41.0 / 32.0 * tmp1)
    z_3 = nu * (8 - 6 * nu)
    d_2 = -6 * nu
    d_3 = tmp0 * (3 * nu - 26)
    v_meco_p2 = 2 * jnp.sqrt(tmp9)
    v_pole_p2 = jnp.sqrt(((1.0 / 3.0) * nu + 1) / (3 - 35.0 / 12.0 * nu))
    F_2 = -35.0 / 12.0 * nu - 1247.0 / 336.0
    F_3 = 4 * pi
    F_4 = (9271.0 / 504.0) * nu + (65.0 / 18.0) * tmp5 - 44711.0 / 9072.0
    F_5 = pi * (-583.0 / 24.0 * nu - 8191.0 / 672.0)
    F_6 = (
        -1712.0 / 105.0 * e_gamma
        + nu
        * (-88.0 / 3.0 * theta_hat + (41.0 / 48.0) * tmp1 - 2913613.0 / 272160.0)
        + (16.0 / 3.0) * tmp1
        - 94403.0 / 3024.0 * tmp5
        - 775.0 / 324.0 * tmp7
        - 856.0 / 105.0 * log(64 * tmp9)
        + 6643739519.0 / 69854400.0
    )
    F_6_l = -856.0 / 105.0
    F_7 = pi * (
        (214745.0 / 1728.0) * nu + (193385.0 / 3024.0) * tmp5 - 16285.0 / 504.0
    )
    f_1 = (55.0 / 42.0) * nu - 43.0 / 21.0
    delta_1_5 = 7.0 / 3.0
    f_2 = -6745.0 / 1512.0 * nu + (2047.0 / 1512.0) * tmp5 - 536.0 / 189.0
    delta_2_5 = -7.0 / 6.0 * nu - 56.0 / 5.0
    delta_3 = (428.0 / 105.0) * pi
    f_3 = (
        -856.0 / 105.0 * e_gamma
        + (41.0 / 96.0) * nu * tmp1
        - 34625.0 / 3696.0 * nu
        - 227875.0 / 33264.0 * tmp5
        + (114635.0 / 99792.0) * tmp7
        - 1712.0 / 105.0 * M_LN2
        + 21428357.0 / 727650.0
    )
    f_3_l = -856.0 / 105.0

    return {
        "e_gamma": e_gamma,
        "pi": pi,
        "theta_hat": theta_hat,
        "a_1": a_1,
        "a_3": a_3,
        "a_4": a_4,
        "z_3": z_3,
        "d_2": d_2,
        "d_3": d_3,
        "v_meco_p2": v_meco_p2,
        "v_pole_p2": v_pole_p2,
        "F_2": F_2,
        "F_3": F_3,
        "F_4": F_4,
        "F_5": F_5,
        "F_6": F_6,
        "F_6_l": F_6_l,
        "F_7": F_7,
        "f_1": f_1,
        "delta_1_5": delta_1_5,
        "f_2": f_2,
        "delta_2_5": delta_2_5,
        "delta_3": delta_3,
        "f_3": f_3,
        "f_3_l": f_3_l,
    }

