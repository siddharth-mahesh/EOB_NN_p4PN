# Create a JAX-compliant complex gamma function
import jax.numpy as jnp

g = 7
n = 9
p = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
]

EPSILON = 1e-07
def gamma(z):
    pi = jnp.pi
    z -= 1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + g + 0.5
    y = jnp.sqrt(2 * pi) * t ** (z + 0.5) * jnp.exp(-t) * x
    return y

if __name__ == "__main__":
    print(gamma(5))
    print(gamma(1 + 1j))