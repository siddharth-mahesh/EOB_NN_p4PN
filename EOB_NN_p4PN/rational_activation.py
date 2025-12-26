import jax
import jax.numpy as jnp
import equinox as eqx

class RationalActivation(eqx.Module):
    """
    A rational activation function with learnable polynomial coefficients.
    
                          1  +   p_1 x + p_2 x^2 + ... + p_n x^n
    Implements: f(x) = -----------------------------------------
                          1  +   q_1 x + q_2 x^2 + ... + q_m x^m
    
    Parameters:
        p: Scale factors (shape: (features, degrees of p - 1))
        q: Frequency/Steepness factors (shape: (features, degrees of q - 1))
    """
    p: jax.Array
    q: jax.Array

    def __init__(self, key: jax.Array, features: int, degree_of_p: int, degree_of_q: int):
        """
        Args:
            key: PRNG key for initialization.
            features: The number of neurons (input dimension).
            degree_of_p: The degree of the numerator polynomial.
            degree_of_q: The degree of the denominator polynomial.
        """
        k_p, k_q = jax.random.split(key, 2)
        self.p = jax.random.normal(k_p, (features,degree_of_p - 1)) * 0.1
        self.q = jax.random.normal(k_q, (features,degree_of_q - 1)) * 0.1

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Applies the activation function element-wise.
        Expected input shape: (..., features)
        """
        x_expanded = x[..., None]
        
        x_powers_p = jnp.power(x_expanded, jnp.arange(1, self.p.shape[-1] + 1))
        numerator = 1.0 + jnp.einsum('...fd, fd -> ...f', x_powers_p, self.p)
        x_powers_q = jnp.power(x_expanded, jnp.arange(1, self.q.shape[-1] + 1))
        denominator = 1.0 + jnp.einsum('...fd, fd -> ...f', x_powers_q, self.q)
        return numerator / denominator


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model_key, data_key = jax.random.split(key)
    activation = RationalActivation(model_key, features=5, degree_of_p=4, degree_of_q=5)
    x_input = jax.random.normal(data_key, (5,))
    print(x_input.shape)
    output = activation(x_input)
    print(output.shape)
    def loss_fn(model, x):
        pred = model(x)
        return jnp.sum(pred ** 2)
    grads = eqx.filter_grad(loss_fn)(activation, x_input)
