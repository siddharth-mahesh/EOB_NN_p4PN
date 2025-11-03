import sympy as sp
from nrpy.c_codegen import c_codegen as ccg

# Basic symbols
nu , r , phi , p_r , p_phi = sp.symbols('nu r phi p_r p_phi',real=True)
a , d , z_3 = sp.symbols('a d z_3',real=True)
u = sp.Symbol('u',real=True)
l_term = p_phi * p_phi * u * u
p_r2 = p_r * p_r
heff = sp.sqrt(a * (1 + p_r2 * (a / d + z_3 * u * u * p_r2) + l_term))
h_real = sp.sqrt(1 + 2 * nu * (heff - 1)) / nu
nrpy_ccode = ccg(h_real, ['const REAL h_real'], include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt').replace('pow', 'jnp.pow')
outstr = f"""
def _hamiltonian(self, y, nu, constants):
    \"\"\"
    Compute the Hamiltonian.

    Args:
        y (jnp.ndarray): Canonical variables [r, phi, p_r, p_phi].
        nu (float): Symmetric mass ratio.
        constants (dict): Dictionary of constants.

    Returns:
        float: Hamiltonian evaluated at y.
    \"\"\"
    r , phi , p_r , p_phi = y
    u = 1 / r
    z_3 = constants['z_3']
    a = self._a_potential(r, constants)
    d = self._d_potential(r, constants)
{nrpy_pycode}    return h_real
"""
with open('hamiltonian.txt','w') as f:
    f.write(outstr)

c = sp.simplify(1/p_r * sp.diff(h_real,p_r))
c_circ = sp.simplify(c.subs(p_r,0))
nrpy_ccode = ccg(c_circ, ['const REAL c_circ'], include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt').replace('pow', 'jnp.pow')
outstr = f"""
def _c_potential(self, r, p_phi, nu, constants):
    \"\"\"
    Compute the Hamiltonian C potential.
    The C potential is given by:
        C = lim_{{p_r -> 0}} (1/p_r * dH/dp_r)

    Args:
        r (float): Radial position.
        p_phi (float): Angular momentum.
        nu (float): Symmetric mass ratio.
        constants (dict): Dictionary of constants.

    Returns:
        float: Hamiltonian C potential.
    \"\"\"
    u = 1 / (r + 1e-100)
    a = self._a_potential(r, constants)
    d = self._d_potential(r, constants)
{nrpy_pycode}    return c_circ
"""
with open('c_potential.txt','w') as f:
    f.write(outstr)

w_circ = sp.simplify(sp.diff(h_real.subs(p_r,0),p_phi))
nrpy_ccode = ccg(w_circ, ['const REAL w_circ'], include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt').replace('pow', 'jnp.pow')
outstr = f"""
def _w_circ(self, r, p_phi, nu, constants):
    \"\"\"
    Compute the circular frequency.

    Args:
        r (float): Radial position.
        p_phi (float): Angular momentum.
        nu (float): Symmetric mass ratio.
        constants (dict): Dictionary of constants.

    Returns:
        float: Circular frequency.
    \"\"\"
    u = 1 / (r + 1e-100)
    a = self._a_potential(r, constants)
    d = self._d_potential(r, constants)
{nrpy_pycode}    return w_circ
"""
with open('w_circ.txt','w') as f:
    f.write(outstr)


pi = sp.Symbol('pi',real=True)
e_gamma = sp.Symbol('e_gamma',real=True)
theta_hat = sp.Symbol('theta_hat',real=True)
unit_tensor = sp.sympify(1)
a_1 = - sp.Rational(2,1) * unit_tensor
a_3 = sp.Rational(2,1) * nu
a_4 = (sp.Rational(94,3) - sp.Rational(41,32)*pi*pi) * nu
z_3 = 2 * (4 - 3 * nu) * nu
d_2 = -6 * nu
d_3 = 2 * nu * (3 * nu - 26)
v_meco_p2 = 2*sp.sqrt((nu + 3)*(-140*nu**3 + 2979*nu**2 - 7956*nu + 2*(nu + 3)*(35*nu - 36)*sp.sqrt(4*nu**2 - 81*nu + 144) + 5184)/((35*nu - 36)*(140*nu**3 - 2979*nu**2 + 7956*nu - 5184)))
v_meco_p2_direct = 2*sp.sqrt((-4*nu**3 + 2*nu**2*sp.sqrt(4*nu**2 - 81*nu + 144) + 69*nu**2 + 12*nu*sp.sqrt(4*nu**2 - 81*nu + 144) + 99*nu + 18*sp.sqrt(4*nu**2 - 81*nu + 144) - 432)/(140*nu**3 - 2979*nu**2 + 7956*nu - 5184))
v_pole_p2 = sp.sqrt((1 + nu / 3) / (3 * (1 - 35 * nu / 36)))
F_2 = - sp.Rational(1247,336) - sp.Rational(35,12)*nu
F_3 = 4 * pi * unit_tensor
F_4 = - sp.Rational(44711,9072) + sp.Rational(9271,504)*nu + sp.Rational(65,18)*nu*nu
F_5 = - (sp.Rational(8191,672) + sp.Rational(583,24)*nu) * pi
F_6_l = - sp.Rational(856,105)*unit_tensor
F_6 = sp.Rational(6643739519,69854400) + sp.Rational(16,3)*pi*pi - sp.Rational(1712,105)*e_gamma + (-sp.Rational(2913613,272160) + sp.Rational(41,48)*pi*pi - sp.Rational(88,3)*theta_hat)*nu - sp.Rational(94403,3024)*nu*nu - sp.Rational(775,324)*nu*nu*nu + F_6_l * sp.log(16 * v_meco_p2**2)
F_7 = (-sp.Rational(16285,504) + sp.Rational(214745,1728)*nu + sp.Rational(193385,3024)*nu*nu) * pi
r0 = 2 / sp.sqrt(sp.E)
f_1 = sp.Rational(55,42)*nu - sp.Rational(43,21)
delta_1_5 = sp.Rational(7,3)
f_2 = sp.Rational(2047,1512)*nu**2 - sp.Rational(6745,1512)*nu - sp.Rational(536,189)
delta_2_5 = -sp.Rational(7,6)*nu - sp.Rational(56,5)
delta_3 = sp.Rational(428,105)*pi
f_3 = -sp.Rational(856,105)*e_gamma + sp.Rational(114635,99792)*nu**3 - sp.Rational(227875,33264)*nu**2 - sp.Rational(34625,3696)*nu + sp.Rational(41,96)*pi**2*nu - sp.Rational(1712,105)*sp.log(2) + sp.Rational(21428357,727650)
f_3_l = - sp.Rational(856,105)
constants = [a_1 , a_3 , a_4 , z_3 , d_2 , d_3 , v_meco_p2 , v_pole_p2 , F_2 , F_3 , F_4 , F_5 , F_6 , F_6_l , F_7 , f_1 , delta_1_5 , f_2 , delta_2_5 , delta_3 , f_3, f_3_l]
constants_labels = ["const REAL a_1" , "const REAL a_3" , "const REAL a_4" , "const REAL z_3" , "const REAL d_2" , "const REAL d_3" , "const REAL v_meco_p2" , "const REAL v_pole_p2" , "const REAL F_2" , "const REAL F_3" , "const REAL F_4" , "const REAL F_5" , "const REAL F_6" , "const REAL F_6_l" , "const REAL F_7" , "const REAL f_1" , "const REAL delta_1_5" , "const REAL f_2" , "const REAL delta_2_5" , "const REAL delta_3" , "const REAL f_3" , "const REAL f_3_l"]

nrpy_ccode = ccg(constants, constants_labels, include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt')
outstr = f"""
def _set_eob_constants_3PN(self, nu):
    \"\"\"
    Calculate the dictionary of EOB constants.

    Args:
        nu (float): Symmetric mass ratio.

    Returns:
        dict: Dictionary of EOB constants.
    \"\"\"
    # All constants are batched with nu so need to define unit_tensor to broadcast constants that are not nu-dependent
    e_gamma = 0.577215664901532860606512090082402431042
    pi = 3.14159265358979323846264338327950288419716939937510
    theta_hat = 1039/4620
    M_LN2 = jnp.log(2)
{nrpy_pycode}
    return {{
        'e_gamma': e_gamma, 'pi': pi, 'theta_hat': theta_hat,
        'a_1': a_1, 'a_3': a_3, 'a_4': a_4, 'z_3': z_3,
        'd_2': d_2, 'd_3': d_3,
        'v_meco_p2': v_meco_p2, 'v_pole_p2': v_pole_p2,
        'F_2': F_2, 'F_3': F_3, 'F_4': F_4, 'F_5': F_5, 'F_6': F_6,
        'F_6_l': F_6_l, 'F_7': F_7,
        'f_1': f_1, 'delta_1_5': delta_1_5, 'f_2': f_2,
        'delta_2_5': delta_2_5, 'delta_3': delta_3, 
        'f_3': f_3, 'f_3_l': f_3_l
    }}
"""

with open('set_eob_constants_3PN.txt','w') as f:
    f.write(outstr)

# write the proper strain expression
# EOB symbols
phi , H , Omega = sp.symbols("phi H Omega",real = True)
# PN symbols
f_1 , f_2 , f_3 , f_3_l , delta_1_5 , delta_2_5 , delta_3 , r0 = sp.symbols("f_1 , f_2 , f_3 , f_3_l , delta_1_5 , delta_2_5 , delta_3 , r0",real=True)
# Constants
Heff_sym = (H**2 - 1)/(2 * nu) + 1
x_sym = Omega**sp.Rational(2,3)
v_sym = Omega**sp.Rational(1,3)
h_N = 8 * nu * x_sym * sp.sqrt(pi*sp.Rational(1,5)) * sp.exp(-sp.I*2*phi)
f22 = 1 + f_1 * x_sym + f_2 * x_sym**2 + (f_3 + f_3_l * sp.log(v_sym))* x_sym**3
delta22 = delta_1_5 * x_sym**sp.Rational(3,2) + delta_2_5 * x_sym**sp.Rational(5,2) + delta_3 * x_sym**3
khat_sym = 2 * Omega * H
gamma = sp.Function("gamma")
T22 = gamma(3 - 2 * sp.I * khat_sym)*sp.exp(pi * khat_sym + 2 * sp.I * khat_sym * sp.log(4 * Omega * r0))/sp.factorial(2)
h22 = h_N * f22 * sp.exp(sp.I * delta22) * T22 * Heff_sym

nrpy_ccode = ccg(h22, 'const REAL h22', include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt').replace('log','jnp.log').replace('exp','jnp.exp').replace('pow','jnp.pow').replace('cbrt','jnp.cbrt')
outstr = f"""
def _strain(self, point , nu , constants):
    \"\"\"
    Calculate the EOB factorized resummed strain.

    Args:
        point (jnp.ndarray): point in the trajectory [r,phi,prstar,pphi].
        nu (float): symmetric mass ratio
        constants (Dict): dictionary of EOB constants

    Returns:
        strain (complex): Complex GW strain.
    \"\"\"
    f_1 = constants['f_1']
    f_2 = constants['f_2']
    f_3 = constants['f_3']
    f_3_l = constants['f_3_l']
    delta_1_5 = constants['delta_1_5']
    delta_2_5 = constants['delta_2_5']
    delta_3 = constants['delta_3']
    # t and r_ISCO don't matter here
    Omega = _eom(0,point,(nu,constants,0))[1]
    H = _hamiltonian(point,nu,constants)
{nrpy_pycode}
    return strain
"""

with open('strain.txt','w') as f:
    f.write(outstr)
