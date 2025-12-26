def pade_1_3(x, a_1, a_3, a_4):
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
