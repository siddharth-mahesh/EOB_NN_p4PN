def pade_0_3(x, d_2, d_3):
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
