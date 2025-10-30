# generate symbolic Padé approximants.
import sympy as sp
from nrpy.c_codegen import c_codegen as ccg
import time

start_time = time.time()
v_p , v , x = sp.symbols('v_p v x',real=True,positive=True)
nu = sp.Symbol("nu",real=True)
eta = nu
e_2 = -x * (1 + sp.Rational(1,3)*eta - (4 - sp.Rational(9,4)*eta + sp.Rational(1,9)*eta**2)*x) / (1 + sp.Rational(1,3)*eta - (3 - sp.Rational(35,12)*eta)*x)
soln = sp.solve(sp.diff(e_2,x),x)
v_meco = sp.sqrt(soln[1])
end_time = time.time()
print(sp.pycode(sp.simplify(v_meco)).replace('math.','sp.'))
print(f"v_MECO solved! Time taken: {end_time - start_time:.2f} seconds")

# Fast P resummation of the GW flux
max_order = 7
p2_order = 5

start_time = time.time()
# Basic symbols
F = [sp.Symbol(f"F_{i}",real=True) for i in range(max_order + 1)]
F_6_l, v_m = sp.symbols("F_6_l v_m",real=True)
F[0] = sp.sympify(1)
F[1] = sp.sympify(0)
c = [sp.Symbol(f"c_{i}",real=True) for i in range(max_order + 1)]
c[0] = sp.sympify(1)

# Full set of inputs: F_i , f_6_l , v , v_p , v_m

# RHS = F_T*(1 - v/v_p) Taylor expanded to v^7 = sum (F_T[i] - F_T[i-1]/v_p)*v**i
rhs = F[0] + sum((F[i] - F[i - 1]/v_p)*v**i for i in range(1,max_order + 1))
rhs_T = [sp.series(rhs,v,0,max_order+1).removeO().coeff(v,i) for i in range(max_order + 1)]
end_time = time.time()
print(f"rhs solved! Time taken: {end_time - start_time:.2f} seconds")
# LHS = 1/(1 + c1 v / (1 + c2 v/ 1 + ...)) Taylor expanded to v^7
start_time = time.time()
lhs = 1 + c[max_order]*v
i = max_order - 1
while i > 0:
    lhs = 1 + c[i] * v / lhs
    i -= 1
lhs = c[0] / lhs
lhs_T = [sp.series(lhs,v,0,max_order+1).removeO().coeff(v,i) for i in range(max_order + 1)]
end_time = time.time()
print(f"lhs solved! Time taken: {end_time - start_time:.2f} seconds")

start_time = time.time()
equations = [lhs_T[i] - rhs_T[i] for i in range(max_order + 1)]

soln = [0]
for i in range(1,max_order+1):
    soln_i = sp.solve(equations[i],c[i])
    c_i_soln = soln_i[0]
    soln.append(c_i_soln)
    #sp.pretty_print(c_i_soln)
end_time = time.time()
print(f"c coefficients solved! Time taken: {end_time - start_time:.2f} seconds")
# Newton-factorized, RR flux at 3.5 PN order
start_time = time.time()
flux_prefactor = (1 + 2 * v**6 * F_6_l * sp.log(v / v_m))/(1 - v/v_p)
cf = 1 + c[max_order]*v
i = max_order - 1
while i > 0:
    cf = 1 + c[i] * v / cf
    i -= 1
cf = c[0] / cf
f_P = flux_prefactor*cf
flux = -sp.Rational(32,5)*nu*v**7*f_P
end_time = time.time()
print(f"flux solved! Time taken: {end_time - start_time:.2f} seconds")

out_str = f"""import sympy as sp
from nrpy.c_codegen import c_codegen as ccg
nu = sp.Symbol('nu',real=True)
F_2 , F_3 , F_4 , F_5 , F_6 , F_7 = sp.symbols('F_2 F_3 F_4 F_5 F_6 F_7',real=True)
F_6_l = sp.symbols('F_6_l',real=True)
v , v_p , v_m = sp.symbols('v v_p v_m',real=True)
c_1 = {sp.pycode(soln[1]).replace('math.','sp.')}
c_2 = {sp.pycode(soln[2]).replace('math.','sp.')}
c_3 = {sp.pycode(soln[3]).replace('math.','sp.')}
c_4 = {sp.pycode(soln[4]).replace('math.','sp.')}
c_5 = {sp.pycode(soln[5]).replace('math.','sp.')}
c_6 = {sp.pycode(soln[6]).replace('math.','sp.')}
c_7 = {sp.pycode(soln[7]).replace('math.','sp.')}
flux = {sp.pycode(flux).replace('math.','sp.').replace('32/5','sp.Rational(32,5)')}
nrpy_ccode = ccg(flux, ['const REAL flux'], include_braces=False, verbose=False)
nrpy_pycode = nrpy_ccode.replace('const REAL ', '    ')
nrpy_pycode = nrpy_pycode.replace(';', '')
nrpy_pycode = nrpy_pycode.replace('sqrt', 'jnp.sqrt').replace('pow', 'jnp.pow').replace('log','jnp.log')
out_str = f\"\"\"
def _flux(self, v, nu, constants):
    \\"\\"\\"
    Compute the circular gravitational flux at the 3.5 PN order

    Args:
        v (float): Orbital velocity
        nu (float): Compactness
        constants (dict): Dictionary of constants

    Returns:
        float: Gravitational flux
    \\"\\"\\"
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
{{nrpy_pycode}}
    return flux
\"\"\"
with open('flux.txt','w') as f:
    f.write(out_str)
"""
with open('flux_v7_presummed.py','w') as outfile:
    outfile.write(out_str)