import os
import sympy as sp
from nrpy.c_codegen import c_codegen as ccg
outfolder = 'codegen_3pn'
nu = sp.Symbol('nu',real=True)
F_2 , F_3 , F_4 , F_5 , F_6 , F_7 = sp.symbols('F_2 F_3 F_4 F_5 F_6 F_7',real=True)
F_6_l = sp.symbols('F_6_l',real=True)
v , v_p , v_m = sp.symbols('v v_p v_m',real=True)
c_1 = 1/v_p
c_2 = F_2/c_1 - c_1
c_3 = F_2/(c_1*c_2*v_p) - F_3/(c_1*c_2) - c_1**2/c_2 - 2*c_1 - c_2
c_4 = -F_3/(c_1*c_2*c_3*v_p) + F_4/(c_1*c_2*c_3) - c_1**3/(c_2*c_3) - 3*c_1**2/c_3 - 3*c_1*c_2/c_3 - 2*c_1 - c_2**2/c_3 - 2*c_2 - c_3
c_5 = F_4/(c_1*c_2*c_3*c_4*v_p) - F_5/(c_1*c_2*c_3*c_4) - c_1**4/(c_2*c_3*c_4) - 4*c_1**3/(c_3*c_4) - 6*c_1**2*c_2/(c_3*c_4) - 3*c_1**2/c_4 - 4*c_1*c_2**2/(c_3*c_4) - 6*c_1*c_2/c_4 - 2*c_1*c_3/c_4 - 2*c_1 - c_2**3/(c_3*c_4) - 3*c_2**2/c_4 - 3*c_2*c_3/c_4 - 2*c_2 - c_3**2/c_4 - 2*c_3 - c_4
c_6 = -F_5/(c_1*c_2*c_3*c_4*c_5*v_p) + F_6/(c_1*c_2*c_3*c_4*c_5) - c_1**5/(c_2*c_3*c_4*c_5) - 5*c_1**4/(c_3*c_4*c_5) - 10*c_1**3*c_2/(c_3*c_4*c_5) - 4*c_1**3/(c_4*c_5) - 10*c_1**2*c_2**2/(c_3*c_4*c_5) - 12*c_1**2*c_2/(c_4*c_5) - 3*c_1**2*c_3/(c_4*c_5) - 3*c_1**2/c_5 - 5*c_1*c_2**3/(c_3*c_4*c_5) - 12*c_1*c_2**2/(c_4*c_5) - 9*c_1*c_2*c_3/(c_4*c_5) - 6*c_1*c_2/c_5 - 2*c_1*c_3**2/(c_4*c_5) - 4*c_1*c_3/c_5 - 2*c_1*c_4/c_5 - 2*c_1 - c_2**4/(c_3*c_4*c_5) - 4*c_2**3/(c_4*c_5) - 6*c_2**2*c_3/(c_4*c_5) - 3*c_2**2/c_5 - 4*c_2*c_3**2/(c_4*c_5) - 6*c_2*c_3/c_5 - 2*c_2*c_4/c_5 - 2*c_2 - c_3**3/(c_4*c_5) - 3*c_3**2/c_5 - 3*c_3*c_4/c_5 - 2*c_3 - c_4**2/c_5 - 2*c_4 - c_5
c_7 = F_6/(c_1*c_2*c_3*c_4*c_5*c_6*v_p) - F_7/(c_1*c_2*c_3*c_4*c_5*c_6) - c_1**6/(c_2*c_3*c_4*c_5*c_6) - 6*c_1**5/(c_3*c_4*c_5*c_6) - 15*c_1**4*c_2/(c_3*c_4*c_5*c_6) - 5*c_1**4/(c_4*c_5*c_6) - 20*c_1**3*c_2**2/(c_3*c_4*c_5*c_6) - 20*c_1**3*c_2/(c_4*c_5*c_6) - 4*c_1**3*c_3/(c_4*c_5*c_6) - 4*c_1**3/(c_5*c_6) - 15*c_1**2*c_2**3/(c_3*c_4*c_5*c_6) - 30*c_1**2*c_2**2/(c_4*c_5*c_6) - 18*c_1**2*c_2*c_3/(c_4*c_5*c_6) - 12*c_1**2*c_2/(c_5*c_6) - 3*c_1**2*c_3**2/(c_4*c_5*c_6) - 6*c_1**2*c_3/(c_5*c_6) - 3*c_1**2*c_4/(c_5*c_6) - 3*c_1**2/c_6 - 6*c_1*c_2**4/(c_3*c_4*c_5*c_6) - 20*c_1*c_2**3/(c_4*c_5*c_6) - 24*c_1*c_2**2*c_3/(c_4*c_5*c_6) - 12*c_1*c_2**2/(c_5*c_6) - 12*c_1*c_2*c_3**2/(c_4*c_5*c_6) - 18*c_1*c_2*c_3/(c_5*c_6) - 6*c_1*c_2*c_4/(c_5*c_6) - 6*c_1*c_2/c_6 - 2*c_1*c_3**3/(c_4*c_5*c_6) - 6*c_1*c_3**2/(c_5*c_6) - 6*c_1*c_3*c_4/(c_5*c_6) - 4*c_1*c_3/c_6 - 2*c_1*c_4**2/(c_5*c_6) - 4*c_1*c_4/c_6 - 2*c_1*c_5/c_6 - 2*c_1 - c_2**5/(c_3*c_4*c_5*c_6) - 5*c_2**4/(c_4*c_5*c_6) - 10*c_2**3*c_3/(c_4*c_5*c_6) - 4*c_2**3/(c_5*c_6) - 10*c_2**2*c_3**2/(c_4*c_5*c_6) - 12*c_2**2*c_3/(c_5*c_6) - 3*c_2**2*c_4/(c_5*c_6) - 3*c_2**2/c_6 - 5*c_2*c_3**3/(c_4*c_5*c_6) - 12*c_2*c_3**2/(c_5*c_6) - 9*c_2*c_3*c_4/(c_5*c_6) - 6*c_2*c_3/c_6 - 2*c_2*c_4**2/(c_5*c_6) - 4*c_2*c_4/c_6 - 2*c_2*c_5/c_6 - 2*c_2 - c_3**4/(c_4*c_5*c_6) - 4*c_3**3/(c_5*c_6) - 6*c_3**2*c_4/(c_5*c_6) - 3*c_3**2/c_6 - 4*c_3*c_4**2/(c_5*c_6) - 6*c_3*c_4/c_6 - 2*c_3*c_5/c_6 - 2*c_3 - c_4**3/(c_5*c_6) - 3*c_4**2/c_6 - 3*c_4*c_5/c_6 - 2*c_4 - c_5**2/c_6 - 2*c_5 - c_6
flux = -sp.Rational(32,5)*nu*v**7*(2*F_6_l*v**6*sp.log(v/v_m) + 1)/((-v/v_p + 1)*(c_1*v/(c_2*v/(c_3*v/(c_4*v/(c_5*v/(c_6*v/(c_7*v + 1) + 1) + 1) + 1) + 1) + 1) + 1))
nrpy_ccode = ccg(flux, ['const REAL flux'], include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt').replace('pow', 'jnp.pow').replace('log','jnp.log')
out_str = f"""
def _flux(self, v, nu, constants):
    \"\"\"
    Compute the circular gravitational flux at the 3.5 PN order

    Args:
        v (float): Orbital velocity
        nu (float): Compactness
        constants (dict): Dictionary of constants

    Returns:
        float: Gravitational flux
    \"\"\"
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
{nrpy_pycode}
    return flux
"""
with open(os.path.join(outfolder,'flux.txt'),'w') as f:
    f.write(out_str)
