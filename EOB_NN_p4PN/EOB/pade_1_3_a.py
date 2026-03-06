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
    tmp2 = ((a_1)*(a_1)*(a_1)) + a_3 - a_4*x
    tmp3 = a_3 + a_4*x
    pade_1_3 = (((a_1)*(a_1)*(a_1)*(a_1))*x + 2*a_1*a_3*x + tmp2)/(-((a_1)*(a_1))*tmp3*((x)*(x)) + a_1*tmp3*x - ((a_3)*(a_3))*((x)*(x)*(x)) + tmp2)
    return pade_1_3