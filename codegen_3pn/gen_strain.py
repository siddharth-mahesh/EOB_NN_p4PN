import sympy as sp

# quantities for calculating the factorized resummed coefficients
x, u, r0, gamma_E, nu = sp.symbols("x u r0 gamma_E nu",real = True,positive=True)

# EOB quantities
# we want to relate the inverse radius u = 1/r to the PN expansion parameter x
# We'll use an ansatz of the form u = x + alpha * x^2 + beta * x^3
# compute the pphi for a circular orbit
# and relate Omega = d Hreal / dpphi = x ** (3/2)
alpha , beta = sp.symbols("alpha beta",real = True)
A = 1 - 2 * u + 2 * nu * u**3 + (sp.Rational(94,3) - sp.Rational(41,32) * sp.pi**2) * nu * u**4
j = sp.sqrt(- sp.diff(A,u) / (sp.diff(u**2 * A,u)))
Heff = sp.sqrt(A * (1 + j**2 * u**2))
Hreal = sp.sqrt(1 + 2 * nu * (Heff - 1))
omega_eob = A * j * u**2 / (Hreal * Heff)
u_ansatz = x + alpha * x**2 + beta * x**3
x_equation = u * sp.simplify(sp.series((omega_eob/u**sp.Rational(3,2))**sp.Rational(2,3), u, 0,3)).removeO() - x
x_equation = sp.series(x_equation.subs(u,u_ansatz),x,0,4).removeO()
x_equations = [x_equation.coeff(x,i) for i in range(2,4)]
soln = []
soln.append(sp.solve(x_equations[0],alpha)[0])
soln.append(sp.solve(x_equations[1].subs(alpha,soln[0]),beta)[0])
u_PN = u_ansatz.subs({alpha:soln[0],beta:soln[1]})
h_eff_PN = sp.series(Heff.subs(u,u_PN),x,0,4).removeO()
h_real_PN = sp.sqrt(1 + 2 * nu * (h_eff_PN - 1))
h_eob_PN = sp.series(h_real_PN.subs(u,u_PN),x,0,4).removeO() - x**sp.Rational(7,2)

# define tail "nought" terms
x0 = 1 / r0
ln_x0_prime = sp.Rational(11,18) - sp.Rational(2,3) * gamma_E - sp.Rational(4,3) * sp.log(2) + sp.Rational(2,3) * sp.log(x0)

# PN expansion of tail terms
# PN expansion of Gamma(l + 1 - 2ikhat) for  calculated using integration calculator
# we expand y to 3 to avoid the 3.5PN term if we set the sympy order to 4. There is no x**3 term in the expansion so this works out.
khat = 2 * x ** sp.Rational(3,2) * h_eob_PN
y = sp.series(2 * sp.I * khat,x,0,3).removeO()
y2 = sp.series(y**2,x,0,4).removeO()
tail_gamma_PN = sp.series(2 + y * (2 * gamma_E - 3) + y2 * sp.Rational(1,2) * (sp.pi**2 * sp.Rational(1,3) + 2 * gamma_E**2 - 6 * gamma_E + 2),x,0,4).removeO()
k = 2 * x ** sp.Rational(3,2)
tail_prefactor = sp.exp(sp.pi * y / (2 * sp.I)) * sp.exp(y * sp.log(2 * k * r0)) / 2
tail_prefactor = sp.series(tail_prefactor,x,0,4).removeO()
tail_PN = sp.series(tail_prefactor * tail_gamma_PN,x,0,4).removeO()

# PN expansion of effective source term
S_eff = h_eff_PN

# residual terms up to 3 PN
f_1 , f_2 , f_3 = sp.symbols("f_1 f_2 f_3",real = True)
delta_1_5 , delta_2_5 , delta_3 = sp.symbols("delta_1_5 delta_2_5 delta_3",real = True)
delta_22 = delta_1_5 * x ** sp.Rational(3,2) + delta_2_5 * x ** sp.Rational(5,2) + delta_3 * x ** 3
f_22 = 1 + f_1 * x + f_2 * x ** 2 + f_3 * x ** 3

newton_normalized_strain = S_eff * tail_PN * f_22 * sp.exp(sp.I * delta_22)

# introduce x = v**2 so that the expansion is in terms of v (desirable for EOB)
v = sp.Symbol("v",real=True,positive=True)
newton_normalized_strain = sp.series(newton_normalized_strain.subs(x,v**2),v,0,7)

# PN expansion of instantaneous amplitudes
H22_0_inst = 1
H22_1_inst = x * (
    - sp.Rational(107,42)
    + sp.Rational(55,42) * nu
)
H22_2_inst = x ** 2 * (
    -sp.Rational(2173,1512)
    - sp.Rational(1069,216) * nu
    + sp.Rational(2047,1512) * nu ** 2
)
H22_2_5_inst = - sp.I * x ** sp.Rational(5,2) * (
    sp.Rational(56,5)
)
H22_3_inst = x ** 3 * (
    sp.Rational(761273,13200)
    + sp.Rational(856,105) * (sp.log(x/x0))
    + (
        sp.Rational(41,96) * sp.pi**2
        - sp.Rational(278185,33264)
    ) * nu
    - sp.Rational(20261,2772) * nu ** 2
    + sp.Rational(114635,99792) * nu ** 3
)

H22_inst = H22_0_inst + H22_1_inst + H22_2_inst + H22_2_5_inst + H22_3_inst

# PN expansion of tail amplitudes
H22_1_5_tail = x ** sp.Rational(3,2) * (
    2 * sp.pi
    + 6 * sp.I * (sp.log(x) - ln_x0_prime)
)
H22_2_5_tail = x ** sp.Rational(5,2) * (
    sp.pi * (
        sp.Rational(34,21) * nu
        - sp.Rational(107,21)
    )
    + sp.I * (sp.log(x) - ln_x0_prime) * (
        sp.Rational(34,7) * nu
        - sp.Rational(107,7)
    )
)
H22_3_tail = x ** 3 * (
    - sp.Rational(515063,22050)
    + sp.Rational(428,105) * sp.I * sp.pi
    + sp.Rational(2,3) * sp.pi**2
    + (sp.log(x) - ln_x0_prime) * (
        12 * sp.I * sp.pi
        - sp.Rational(428,35)
    )
    - 18 * (sp.log(x) - ln_x0_prime)**2
)

H22_tail = H22_1_5_tail + H22_2_5_tail + H22_3_tail

H22 = sp.series((H22_inst + H22_tail).subs(x,v**2),v,0,7).removeO()

# perform the factorized resummed computation for the strain
# The computation is done by matching the tailor expanded terms of the factorized strain to the PN expanded strains.
# If we do this by order, we end up with a LOT of trivial equations. It is better to list the orders that matter and extract only those coefficients:
# PN order | re/im    | unknown
#  1       | real     |    f_1
#  1.5     | imag     | delta_1_5
#  2       | real     |    f_2
#  2.5     | imag     | delta_2_5
#  3       | real     |    f_3
#  3       | imag     | delta_3

subs_dict = {}

# 1PN real equation
lhs = sp.re(newton_normalized_strain.coeff(v,2))
rhs = sp.re(H22.coeff(v,2))
equation = sp.Eq(sp.simplify(lhs-rhs),sp.sympify(0))
soln = sp.solve(equation,f_1)[0]
subs_dict.update({f_1:soln})

# 1.5PN imaginary equation
lhs = sp.im(newton_normalized_strain.coeff(v,3)).subs(subs_dict)
rhs = sp.im(H22.coeff(v,3)).subs(subs_dict)
equation = sp.Eq(sp.simplify(lhs-rhs),sp.sympify(0))
soln = sp.solve(equation,delta_1_5)[0]
subs_dict.update({delta_1_5:soln})

# 2PN real equation
lhs = sp.re(newton_normalized_strain.coeff(v,4)).subs(subs_dict)
rhs = sp.re(H22.coeff(v,4)).subs(subs_dict)
equation = sp.Eq(sp.simplify(lhs-rhs),sp.sympify(0))
soln = sp.solve(equation,f_2)[0]
subs_dict.update({f_2:soln})

# 2.5PN imaginary equation
lhs = sp.im(newton_normalized_strain.coeff(v,5)).subs(subs_dict)
rhs = sp.im(H22.coeff(v,5)).subs(subs_dict)
equation = sp.Eq(sp.simplify(lhs-rhs),sp.sympify(0))
soln = sp.solve(equation,delta_2_5)[0]
subs_dict.update({delta_2_5:soln})

# 3PN imaginary equation
lhs = sp.im(newton_normalized_strain.coeff(v,6)).subs(subs_dict)
rhs = sp.im(H22.coeff(v,6)).subs(subs_dict)
equation = sp.Eq(sp.simplify(lhs-rhs),sp.sympify(0))
soln = sp.solve(equation,delta_3)[0]
subs_dict.update({delta_3:soln})

# 3PN real equation
lhs = sp.re(newton_normalized_strain.coeff(v,6)).subs(subs_dict)
rhs = sp.re(H22.coeff(v,6))
equation = sp.Eq(lhs-rhs,sp.sympify(0))
soln = sp.simplify(sp.solve(equation,f_3)[0])
subs_dict.update({f_3:soln})
for key , value in subs_dict.items():
    print(f"{key} = {value}")