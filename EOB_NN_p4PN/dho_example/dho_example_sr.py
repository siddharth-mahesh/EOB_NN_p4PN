import time
import tempfile
import equinox as eqx
import diffrax
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
#jax.config.update("jax_enable_x64", True)
import optax
import pysr
import sympy
import sympy2jax
import matplotlib.pyplot as plt


class Func(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        # Best practice is often to use `learnt_scalar * tanh(MLP(...))` for the
        # vector field.
        return self.out_scale * self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

def _get_data(ts, *, key):
    y0 = jr.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        return jnp.stack([y[1], -y[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def run_neural_ode(
    dataset_size=256,
    batch_size=32,
    lr=3e-3,
    steps_strategy=(500, 500),
    length_strategy=(0.1, 0.1),
    width_size=64,
    depth=2,
    seed=5678,
    plot=True,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape

    model = NeuralODE(data_size, width_size, depth, key=model_key)
    optim = optax.adabelief(lr)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for steps, length in zip(steps_strategy, length_strategy):
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")

    if plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        model_y = model(ts, ys[0, 0])
        plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
        plt.plot(ts, model_y[:, 1], c="crimson")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neural_ode.png")
        plt.show()

    return ts, ys, model

class Stack(eqx.Module):
    modules: list[eqx.Module]

    def __call__(self, x):
        assert x.shape[-1] == 2
        x0 = x[..., 0]
        x1 = x[..., 1]
        return jnp.stack([module(x0=x0, x1=x1) for module in self.modules], axis=-1)


def quantise(expr, quantise_to):
    if isinstance(expr, sympy.Float):
        return expr.func(round(float(expr) / quantise_to) * quantise_to)
    elif isinstance(expr, (sympy.Symbol, sympy.Integer)):
        return expr
    else:
        return expr.func(*[quantise(arg, quantise_to) for arg in expr.args])

def main(
    symbolic_dataset_size=2000,
    symbolic_num_populations=100,
    symbolic_population_size=20,
    symbolic_migration_steps=4,
    symbolic_mutation_steps=30,
    symbolic_descent_steps=50,
    pareto_coefficient=2,
    fine_tuning_steps=500,
    fine_tuning_lr=3e-3,
    quantise_to=0.01,
):
    #
    # First obtain a neural approximation to the dynamics.
    # We begin by running the previous example.
    #

    # Runs the Neural ODE example.
    # This defines the variables `ts`, `ys`, `model`.
    print("Training neural differential equation.")
    ts, ys, model = run_neural_ode()

    #
    # Now symbolically regress across the learnt vector field, to obtain a Pareto
    # frontier of symbolic equations, that trades loss against complexity of the
    # equation. Select the "best" from this frontier.
    #

    print("Symbolically regressing across the vector field.")
    vector_field = model.func.mlp  # noqa: F821
    dataset_size, length_size, data_size = ys.shape  # noqa: F821
    in_ = ys.reshape(dataset_size * length_size, data_size)  # noqa: F821
    in_ = in_[:symbolic_dataset_size]
    out = jax.vmap(vector_field)(in_)
    with tempfile.TemporaryDirectory() as tempdir:
        symbolic_regressor = pysr.PySRRegressor(
            niterations=symbolic_migration_steps,
            ncycles_per_iteration=symbolic_mutation_steps,
            populations=symbolic_num_populations,
            population_size=symbolic_population_size,
            optimizer_iterations=symbolic_descent_steps,
            optimizer_nrestarts=1,
            procs=1,
            model_selection="score",
            progress=False,
            tempdir=tempdir,
            temp_equation_file=True,
        )
        symbolic_regressor.fit(in_, out)
        best_expressions = [b.sympy_format for b in symbolic_regressor.get_best()]

    #
    # Now the constants in this expression have been optimised for regressing across
    # the neural vector field. This was good enough to obtain the symbolic expression,
    # but won't quite be perfect -- some of the constants will be slightly off.
    #
    # To fix this we now plug our symbolic function back into the original dataset
    # and apply gradient descent.
    #

    print("\nOptimising symbolic expression.")

    symbolic_fn = Stack([sympy2jax.SymbolicModule(expr) for expr in best_expressions])
    symbolic_model = eqx.tree_at(lambda m: m.func.mlp, model, symbolic_fn)  # noqa: F821

    @eqx.filter_grad
    def grad_loss(symbolic_model):
        vmap_model = jax.vmap(symbolic_model, in_axes=(None, 0))
        pred_ys = vmap_model(ts, ys[:, 0])  # noqa: F821
        return jnp.mean((ys - pred_ys) ** 2)  # noqa: F821

    optim = optax.adam(fine_tuning_lr)
    opt_state = optim.init(eqx.filter(symbolic_model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(symbolic_model, opt_state):
        grads = grad_loss(symbolic_model)
        updates, opt_state = optim.update(grads, opt_state)
        symbolic_model = eqx.apply_updates(symbolic_model, updates)
        return symbolic_model, opt_state

    for _ in range(fine_tuning_steps):
        symbolic_model, opt_state = make_step(symbolic_model, opt_state)

    #
    # Finally we round each constant to the nearest multiple of `quantise_to`.
    #

    trained_expressions = []
    for symbolic_module in symbolic_model.func.mlp.modules:
        expression = symbolic_module.sympy()
        expression = quantise(expression, quantise_to)
        trained_expressions.append(expression)

    print(f"Expressions found: {trained_expressions}")

main()