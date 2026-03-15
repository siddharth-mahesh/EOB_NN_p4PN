import sympy as sp
from sympy.plotting import plot
import time

Omega , a = sp.symbols('Omega a',real=True)
a_series = [-0.3 , -0.1 , 0 , 0.1 , 0.3 ]

r = (1/Omega - a)**(sp.Rational(2,3))
L = sp.simplify(
    (r**2 - 2 * a * sp.sqrt(r) + a**2) 
    / 
    (r**(sp.Rational(3,4)) * sp.sqrt(r**(sp.Rational(3,2)) - 3 * sp.sqrt(r) + 2 * a))
)

dL_dOmega = sp.diff(L, Omega)
print(sp.simplify(dL_dOmega))
r_ph = 2 * (1 + sp.cos(sp.Rational(2,3) * sp.acos(-a)))
Omega_ph = 1 / (r_ph ** sp.Rational(3,2) + a)

Z_1 = 1 + sp.cbrt(1 - a**2) * (sp.cbrt(1 + a) + sp.cbrt(1 - a))
Z_2 = sp.sqrt(3 * a**2 + Z_1**2)
r_ms = 3 + Z_2 - sp.sqrt((3 - Z_1) * (3 + Z_1 + 2 * Z_2))
Omega_ms = 1 / (r_ms ** sp.Rational(3,2) + a)

#print(Omega_ph)
#print(Omega_ms)
plot_dL_Omega = False
if plot_dL_Omega:
    p1 = plot(dL_dOmega.subs(a, a_series[0]),(Omega, Omega_ms.subs(a, a_series[0]),0.9*Omega_ph.subs(a, a_series[0])))
    for i in range(1,len(a_series)):
        p1.extend(plot(dL_dOmega.subs(a, a_series[i]),(Omega, Omega_ms.subs(a, a_series[i]),0.9*Omega_ph.subs(a, a_series[i]))))
    p1.save("dL_dOmega.png")
    p1.show()

T = sp.Symbol('T',real=True)
Omega_T = sp.sqrt((Omega_ph**2 - Omega_ms **2) / 2 + sp.tanh(T) * (Omega_ph**2 + Omega_ms**2) / 2)
dL_dOmega_T = (dL_dOmega.subs(Omega, Omega_T))

plot_dL_T = True
if plot_dL_T:
    p1 = plot(dL_dOmega_T.subs(a, a_series[0]),(T, -20, 20))
    for i in range(1,len(a_series)):
        p1.extend(plot(dL_dOmega_T.subs(a, a_series[i]),(T, -20, 20)))
    p1.save("dL_dOmega_T.png")
