import sys
import jax
import jax.numpy as jnp
import equinox as eqx
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module import Neural_EOB
import diffrax
import optimistix

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(0)
model = Neural_EOB(key=key)

x_train = jnp.load("EOB_NN_p4PN/EOB_NNp4PN/x_sxs_1em4.npy")
real_idx = 42
x_sample = x_train[real_idx]
nu = x_sample[0]
constants = model._set_eob_constants_3PN(nu)
ics = model.eob3pn._initial_conditions(x_sample)
ics = jax.lax.stop_gradient(ics)

def solve_with_margin(margin):
    r_LR = jax.lax.stop_gradient(
        optimistix.root_find(
            model._lr_condition, optimistix.Newton(1e-8, 1e-8), 3.0, (nu, constants)
        ).value
    )
    
    def dynamics_with_margin(_margin):
        r_fin = r_LR + _margin
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(model._eom),
            diffrax.Dopri5(),
            t0=0,
            t1=500000.0,
            dt0=0.1,
            y0=ics,
            args=(nu, r_fin, constants),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            event=diffrax.Event(
                model._event_fn, optimistix.Newton(1e-5, 1e-5, optimistix.rms_norm)
            ),
            saveat=diffrax.SaveAt(t0=True, t1=True, dense=True),
            max_steps=100000,
            throw=False,
        )
        
        point = sol.evaluate(sol.ts[-1] - 1.0)
        return jnp.sum(jnp.abs(point)**2)
    
    val, grad = eqx.filter_value_and_grad(dynamics_with_margin)(margin)
    return val, grad

for m in [0.0, 0.01, 0.05, 0.1]:
    try:
        val, grad_m = solve_with_margin(m)
        print(f"Margin {m}M -> Evaluates natively! No NaN. Grad of margin = {grad_m}")
    except Exception as e:
        print(f"Margin {m}M -> Failed: {e}")
