# Example module to test diffrax

import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx 

def vector_field(t, y, args):
    _, v = y
    return jnp.array([v, -8.0])

def cond_fn(t, y, args, **kwargs):
    x, _ = y
    return x

y0 = jnp.array([10.0, 0.0])
t0 = 0
t1 = jnp.inf
dt0 = 0.1
term = diffrax.ODETerm(vector_field)
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = diffrax.Event(cond_fn, root_finder)
solver = diffrax.Tsit5()
sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event, saveat=diffrax.SaveAt(t0=True,t1=True,dense=True))
print(sol)
print(f"Event time: {sol.ts[-1]}") # Event time: 1.58...
print(f"Velocity at event time: {sol.ys[-1, 1]}") # Velocity at event time: -12.64...
times = jnp.linspace(0, sol.ts[-1], 100)
trajectory = jax.vmap(sol.evaluate, in_axes=0)(jnp.array([0.,sol.ts[-1]/2,sol.ts[-1]]))
print(trajectory)
