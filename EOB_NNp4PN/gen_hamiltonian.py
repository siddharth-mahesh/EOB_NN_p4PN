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
nrpy_pycode = nrpy_pycode.replace('sqrt', 'tf.math.sqrt').replace('pow', 'tf.math.pow')
outstr = f"""@tf.function
def _hamiltonian(self, x, nu, constants):
    \"\"\"
    Compute the Hamiltonian.

    Args:
        x (tf.Tensor): Canonical variables [r, phi, p_r, p_phi] (batch_size,4).
        nu (tf.Tensor): Symmetric mass ratio (batch_size,).
        constants (dict): Dictionary of constants.

    Returns:
        tf.Tensor: Hamiltonian evaluated at x (batch_size,1).
    \"\"\"
    r , phi , p_r , p_phi = tf.split(x,4,axis=-1)
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
nrpy_pycode = nrpy_pycode.replace('sqrt', 'tf.math.sqrt').replace('pow', 'tf.math.pow')
outstr = f"""@tf.function
def _c_potential(self, r, p_phi, nu, constants):
    \"\"\"
    Compute the Hamiltonian C potential.
    The C potential is given by:
        C = lim_{{p_r -> 0}} (1/p_r * dH/dp_r)

    Args:
        r (tf.Tensor): Radial position (batch_size,)
        p_phi (tf.Tensor): Angular momentum (batch_size,)
        nu (tf.Tensor): Symmetric mass ratio (batch_size,)
        constants (dict): Dictionary of constants.

    Returns:
        tf.Tensor: Hamiltonian C potential (batch_size,)
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
nrpy_pycode = nrpy_pycode.replace('sqrt', 'tf.math.sqrt').replace('pow', 'tf.math.pow')
outstr = f"""@tf.function
def _w_circ(self, r, p_phi, nu, constants):
    \"\"\"
    Compute the circular frequency.

    Args:
        r (tf.Tensor): Radial position (batch_size,)
        p_phi (tf.Tensor): Angular momentum (batch_size,)
        nu (tf.Tensor): Symmetric mass ratio (batch_size,)
        constants (dict): Dictionary of constants.

    Returns:
        tf.Tensor: Circular frequency (batch_size,)
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
unit_tensor = sp.Symbol('unit_tensor',real=True)
a_1 = - sp.Rational(2,1) * unit_tensor
a_3 = sp.Rational(2,1) * nu
a_4 = (sp.Rational(94,3) - sp.Rational(41,32)*pi*pi) * nu
z_3 = 2 * (4 - 3 * nu) * nu
d_2 = -6 * nu
d_3 = 2 * nu * (3 * nu - 26)
v_meco_p2 = 2*sp.sqrt((nu + 3)*(-140*nu**3 + 2979*nu**2 - 7956*nu + 2*(nu + 3)*(35*nu - 36)*sp.sqrt(4*nu**2 - 81*nu + 144) + 5184)/((35*nu - 36)*(140*nu**3 - 2979*nu**2 + 7956*nu - 5184)))
v_pole_p2 = sp.sqrt((1 + nu / 3) / (3 * (1 - 35 * nu / 36)))
F_2 = - sp.Rational(1247,336) - sp.Rational(35,12)*nu
F_3 = 4 * pi * unit_tensor
F_4 = - sp.Rational(44711,9072) + sp.Rational(9271,504)*nu + sp.Rational(65,18)*nu*nu
F_5 = - (sp.Rational(8191,672) + sp.Rational(583,24)*nu) * pi
F_6 = sp.Rational(6643739519,69854400) + sp.Rational(16,3)*pi*pi - sp.Rational(1712,105)*e_gamma + (-sp.Rational(2913613,272160) + sp.Rational(41,48)*pi*pi - sp.Rational(88,3)*theta_hat)*nu - sp.Rational(94403,3024)*nu*nu - sp.Rational(775,324)*nu*nu*nu
F_6_l = - sp.Rational(856,105)*unit_tensor
F_7 = (-sp.Rational(16285,504) + sp.Rational(214745,1728)*nu + sp.Rational(193385,3024)*nu*nu) * pi
constants = [a_1 , a_3 , a_4 , z_3 , d_2 , d_3 , v_meco_p2 , v_pole_p2 , F_2 , F_3 , F_4 , F_5 , F_6 , F_6_l , F_7]
constants_labels = ["const REAL a_1" , "const REAL a_3" , "const REAL a_4" , "const REAL z_3" , "const REAL d_2" , "const REAL d_3" , "const REAL v_meco_p2" , "const REAL v_pole_p2" , "const REAL F_2" , "const REAL F_3" , "const REAL F_4" , "const REAL F_5" , "const REAL F_6" , "const REAL F_6_l" , "const REAL F_7"]

nrpy_ccode = ccg(constants, constants_labels, include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'tf.math.sqrt')
outstr = f"""@tf.function
def _set_eob_constants_3PN(self, nu):
    \"\"\"
    Calculate the dictionary of EOB constants.

    Args:
        nu (tf.Tensor): Symmetric mass ratio (None,).

    Returns:
        dict: Dictionary of EOB constants.
    \"\"\"
    # All constants are batched with nu so need to define unit_tensor to broadcast constants that are not nu-dependent
    unit_tensor = tf.ones_like(nu)
    e_gamma = tf.constant(0.577215664901532860606512090082402431042, dtype=tf.float64)
    pi = tf.constant(3.14159265358979323846264338327950288419716939937510, dtype=tf.float64)
    theta_hat = tf.constant(1039/4620, dtype=tf.float64)
{nrpy_pycode}    # Cast all constants to tf.float64
    return {{
        'e_gamma': e_gamma, 'pi': pi, 'theta_hat': theta_hat,
        'a_1': tf.cast(a_1, dtype=tf.float64), 'a_3': tf.cast(a_3, dtype=tf.float64), 'a_4': tf.cast(a_4, dtype=tf.float64), 'z_3': tf.cast(z_3, dtype=tf.float64),
        'd_2': tf.cast(d_2, dtype=tf.float64), 'd_3': tf.cast(d_3, dtype=tf.float64),
        'v_meco_p2': tf.cast(v_meco_p2, dtype=tf.float64), 'v_pole_p2': tf.cast(v_pole_p2, dtype=tf.float64),
        'F_2': tf.cast(F_2, dtype=tf.float64), 'F_3': tf.cast(F_3, dtype=tf.float64), 'F_4': tf.cast(F_4, dtype=tf.float64), 'F_5': tf.cast(F_5, dtype=tf.float64), 'F_6': tf.cast(F_6, dtype=tf.float64),
        'F_6_l': tf.cast(F_6_l, dtype=tf.float64), 'F_7': tf.cast(F_7, dtype=tf.float64)
    }}
"""

with open('set_eob_constants_3PN.txt','w') as f:
    f.write(outstr)