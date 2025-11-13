import os
import sympy as sp 
from nrpy.c_codegen import c_codegen as ccg
outfolder = 'codegen_3pn'
a_0 , a_1 , a_3 , a_4 , a_5 , x = sp.symbols('a_0 a_1 a_3 a_4 a_5 x',real = True)
pade_1_4 = (
    (
        (
            (
                a_1**5 
                + 3*a_3*a_1**2 
                - 2*a_4*a_1 
                + a_5
            ) * x
        )/(
            a_1**4 
            + 2*a_3*a_1 
            - a_4
        ) 
        + 1
    )/(
        (
            (
                -a_5 * a_1**3 
                + a_3**2 * a_1**2 
                - 2 * a_3 * a_4 * a_1 
                + a_4**2 
                - a_3 * a_5
            ) * x**4
        )/(
            a_1**4 
            + 2 * a_3 * a_1 
            - a_4
        ) 
        + (
            (
                -a_4 * a_1**3 
                + a_5 * a_1**2 
                - 2 * a_3**2 * a_1 
                + a_3 * a_4
            ) * x**3
        )/(
            a_1**4 
            + 2 * a_3 * a_1 
            - a_4
        ) 
        - (
            a_1 * (
                a_3 * a_1**2 
                - a_4 * a_1 
                + a_5
            ) * x**2
        )/(
            a_1**4 
            + 2 * a_3 * a_1 
            - a_4
        ) 
        + (
            (
                a_3 * a_1**2 
                - a_4 * a_1 
                + a_5
            ) * x
        )/(
            a_1**4 
            + 2 * a_3 * a_1 
            - a_4
        ) 
        + 1
    )
)

nrpy_ccode = ccg(pade_1_4, 'pade_1_4',include_braces=False,verbose=False)
nrpy_pycode = f"""
def pade_1_4(self,a_1, a_3, a_4, a_5, x):
    \"\"\"
    Compute the Pade approximant P^{1}_{4} for the Hamiltonian A potential.
    The Hamiltonian A potential is given by a polynomial of the form
    p(x) = 1 + a_1 x + a_3 x^3 + a_4 x^4 + a_5 x^5

    Args:
        x (float): Input tensor, typically 1/r.
        a_1 (float): Coefficient a_1.
        a_3 (float): Coefficient a_3.
        a_4 (float): Coefficient a_4.
        a_5 (float): Coefficient a_5.

    Returns:
        float: Pade approximant P^{1}_{4} evaluated at x.
    \"\"\"
{nrpy_ccode.replace(
    'const REAL ', '    '
).replace(
    ';', ''
).replace(
    'pade_1_4', '    pade_1_4'
)}    return pade_1_4
"""
with open(os.path.join(outfolder,'pade_1_4.txt'), 'w') as f:
    f.write(nrpy_pycode)

pade_1_3 = (
    (
        (
            (
                a_1**4 
                + 2 * a_3 * a_1 
                - a_4
            ) * x
        )/(
            a_1**3 
            + a_3
        ) 
        + 1
    )/(
        (
            (
                -a_4 * a_1**2 
                - a_3**2
            ) * x**3
        )/(
            a_1**3 
            + a_3
        ) 
        - (
            a_1 * (
                a_1 * a_3 
                - a_4
            ) * x**2
        )/(
            a_1**3 
            + a_3
        ) 
        + (
            (
                a_1 * a_3 
                - a_4
            ) * x
        )/(
            a_1**3 
            + a_3
        ) 
        + 1
    )
)

pade_1_3_ccode = ccg(pade_1_3, 'pade_1_3',include_braces=False,verbose=False)
pade_1_3_pycode = f"""
def pade_1_3(self,x,a_1, a_3, a_4):
    \"\"\"
    Compute the Pade approximant P^{1}_{3} for the Hamiltonian A potential.
    The Hamiltonian A potential is given by a polynomial of the form
    p(x) = 1 + a_1 x + a_3 x^3 + a_4 x^4

    Args:
        x (float): Input tensor, typically 1/r.
        a_1 (float): Coefficient a_1.
        a_3 (float): Coefficient a_3.
        a_4 (float): Coefficient a_4.

    Returns:
        float: Pade approximant P^{1}_{3} evaluated at x.
    \"\"\"
{pade_1_3_ccode.replace(
    'const REAL ', '    '
).replace(
    ';', ''
).replace(
    'pade_1_3', '    pade_1_3'
)}    return pade_1_3
"""
with open(os.path.join(outfolder,'pade_1_3.txt'), 'w') as f:
    f.write(pade_1_3_pycode)

