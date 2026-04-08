r"""Symbolic Regression of Damped Oscillator Systems.

Author: Siddharth Mahesh
sm0193 at mix dot wvu dot edu

This module generates trajectories and equations of motion
for two damped oscillatory systems:

1. The damped harmonic oscillator

\dot{x} = p
\dot{p} = -x - \beta p

2. The damped pendulum

\dot{x} = p
\dot{p} = - \sin{x} - \beta p

The module performs symbolic regression on the generated data
to arrive at the functional form of these systems.
The representations are plotted for accuracy comparisons
and stored in a SymbolicODENetwork class that can be used for model deployment.
"""

import tempfile
from typing import Any, List, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pysr
import sympy
import sympy2jax

# jax.config.update("jax_enable_x64", True)


class SymbolicODENetwork(eqx.Module):
    """A native Symbolic ODE architecture with optimized symbolic RHS expressions."""

    symbolic_rhs: List[Any]

    def __init__(self, best_expressions: List[sympy.Expr]) -> None:
        """Initialise the SymbolicODENetwork with best expressions found by PySR.

        Args:
            best_expressions: A list of the best expressions found by PySR.
        """
        self.symbolic_rhs = [
            sympy2jax.SymbolicModule(expr) for expr in best_expressions
        ]

    def __call__(self, _t: jax.Array, ins: jax.Array) -> jax.Array:
        """Compute the right-hand side of the ODE.

        Args:
            _t: The current time.
            ins: The current state of the system.

        Returns:
            The right-hand side of the ODE.
        """
        # Maps y[0], y[1]... to x0, x1... in the symbolic module
        # Combine y and args into a single vector
        kwargs = {f"x{i}": ins[i] for i in range(ins.shape[0])}
        results = [mod(**kwargs) for mod in self.symbolic_rhs]
        return jnp.stack(results)

    def _ode_parse(self, _t: jax.Array, y: jax.Array, damping: jax.Array) -> jax.Array:
        """Parse the ODE for Diffrax.

        Args:
            _t: The current time.
            y: The current state of the system.
            damping: The damping coefficient.

        Returns:
            The right-hand side of the ODE.
        """
        ins = jnp.hstack([y, damping])
        return self(_t, ins)

    def solve(self, ts: jax.Array, ins: jax.Array) -> jax.Array:
        """Solve the ODE using the symbolic right-hand side.

        Args:
            ts: The time points at which to solve the ODE.
            ins: The initial state of the system.

        Returns:
            The solution to the ODE.
        """
        y0, beta = ins[:2], jnp.array([ins[2]])
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self._ode_parse),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.1,
            y0=y0,
            args=beta,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys  # type: ignore


class DataLoader:
    """A dataloader for the DHO system."""

    def __init__(self, physical_system: str = "damped oscillator") -> None:
        self.allowed_systems: List[str] = ["damped oscillator", "damped pendulum"]
        self.physical_system: str = physical_system
        if physical_system == "damped oscillator":
            self._f = self._f_damped_sho
        elif physical_system == "damped pendulum":
            self._f = self._f_damped_pendulum
        else:
            raise ValueError(
                f"Unknown physical system: {physical_system}. "
                f"Allowed systems: {self.allowed_systems}"
            )

    def _f_damped_sho(self, _t: jax.Array, y: jax.Array, args: jax.Array) -> jax.Array:
        """Evaluate the RHS of the Damped Simple Harmonic Oscillator system.

        Args:
            _t: The current time.
            y: The current state of the system.
            args: The damping coefficient.

        Returns:
            The right-hand side of the ODE.
        """
        return jnp.stack([y[1], -y[0] - args[0] * y[1]], axis=-1)

    def _f_damped_pendulum(self, _t: jax.Array, y: jax.Array, args: jax.Array) -> jax.Array:
        """Evaluate the RHS of the Damped Pendulum system.

        Args:
            _t: The current time.
            y: The current state of the system.
            args: The damping coefficient.

        Returns:
            The right-hand side of the ODE.
        """
        return jnp.stack([y[1], -jnp.sin(y[0]) - args[0] * y[1]], axis=-1)

    def get_data_single(
        self, ts: jax.Array, key: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Generate data for the DHO system for a single random initial condition.

        Args:
            ts: The time points at which to generate the data.
            key: The random key to use for generating the data.

        Returns:
            The data generated by the system.

        Raises:
            ValueError: If the physical system is not recognized.
        """
        key, subkey1, subkey2 = jr.split(key, 3)
        if self.physical_system == "damped oscillator":
            # DHO initial conditions
            # y0 can be anything because the conservative system is linear and therefore dimensionless
            # require beta between 2 and 6 for the system to be underdamped, critically damped, or overdamped
            y0 = jr.uniform(subkey1, (2,), minval=-0.6, maxval=1)
            beta = jr.uniform(subkey2, (1,), minval=2, maxval=6)
        elif self.physical_system == "damped pendulum":
            # Pendulum initial conditions
            # Require theta = y[0] to be between -\pi and \pi
            # Require v = y[1] to be between -5 and 5
            # require beta = y[2] to be between 0.1 to 3. but sample log uniform
            y0_theta = jr.uniform(subkey1, (1,), minval=-jnp.pi, maxval=jnp.pi)
            y0_v = jr.uniform(subkey1, (1,), minval=-5, maxval=5)
            y0 = jnp.hstack([y0_theta, y0_v])
            beta = 10 ** jr.uniform(
                subkey2, (1,), minval=jnp.log10(0.1), maxval=jnp.log10(3)
            )
        else:
            raise ValueError(f"Unknown physical system: {self.physical_system}")

        solver = diffrax.Tsit5()
        dt0 = 0.1
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._f),
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            args=beta,
            saveat=saveat,
        )
        ys = sol.ys
        betas = jnp.tile(beta, (ts.shape[0], 1))
        x_vals = jnp.hstack([ys, betas])
        y_vals = jax.vmap(self._f, in_axes=(None, 0, 0))(ts, ys, betas)
        return x_vals, y_vals

    def __call__(self, dataset_size: int, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Vectorize get_data_single for a given dataset size.

        Args:
            dataset_size: The number of data points to generate.
            key: The random key to use for generating the data.

        Returns:
            The data generated by the DHO system.
        """
        ts = jnp.linspace(0, 10, 100)
        key = jr.split(key, dataset_size)
        xvals, yvals = jax.vmap(lambda key: self.get_data_single(ts, key=key))(key)
        # flatten the first two dimensions
        xvals = xvals.reshape(-1, xvals.shape[-1])
        yvals = yvals.reshape(-1, yvals.shape[-1])
        return xvals, yvals


def quantise(expr: sympy.Expr, quantise_to: float) -> sympy.Expr:
    """Quantize the symbolic expression to a given precision.

    Args:
        expr: The symbolic expression to quantize.
        quantise_to: The precision to quantize to.

    Returns:
        The quantized symbolic expression.
    """
    if isinstance(expr, sympy.Float):
        return expr.func(round(float(expr) / quantise_to) * quantise_to)
    if isinstance(expr, (sympy.Symbol, sympy.Integer)):
        return expr
    return expr.func(*[quantise(arg, quantise_to) for arg in expr.args])


def main(
    physical_system: str = "damped pendulum",
    symbolic_dataset_size: int = 2000,
    symbolic_num_populations: int = 100,
    symbolic_population_size: int = 20,
    symbolic_migration_steps: int = 4,
    symbolic_mutation_steps: int = 30,
    symbolic_descent_steps: int = 50,
    quantise_to: float = 0.01,
) -> None:
    """Run the symbolic regression pipeline.

    Args:
        physical_system: The type of physical system to simulate.
        symbolic_dataset_size: The number of data points to use for symbolic regression.
        symbolic_num_populations: The number of populations to use for symbolic regression.
        symbolic_population_size: The size of each population in symbolic regression.
        symbolic_migration_steps: The number of migration steps in symbolic regression.
        symbolic_mutation_steps: The number of mutation steps in symbolic regression.
        symbolic_descent_steps: The number of descent steps in symbolic regression.
        quantise_to: The precision to quantize the symbolic expressions to.
    """
    # Get the trajectory data
    print("Generating dataset.")
    key = jr.PRNGKey(5678)
    data_key, key = jr.split(key, 2)
    loader = DataLoader(physical_system=physical_system)
    in_, out = loader(dataset_size=symbolic_dataset_size // 100 + 1, key=data_key)

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
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin"],
        )
        symbolic_regressor.fit(in_, out)
        best_expressions = [b.sympy_format for b in symbolic_regressor.get_best()]

    # Quantize the expressions
    quantized_expressions = [quantise(expr, quantise_to) for expr in best_expressions]

    print(f"Initial expressions found by PySR: {best_expressions}")
    print(f"Quantized expressions: {quantized_expressions}")
    # Use the new SymbolicODENetwork architecture
    symbolic_model = SymbolicODENetwork(quantized_expressions)

    # Plot initial comparison (before fine-tuning)
    ts = jnp.linspace(0, 10, 100)
    traj_key, key = jr.split(key, 2)
    x_traj, _ = loader.get_data_single(ts, key=traj_key)
    plt.figure(figsize=(12, 5))
    plt.plot(ts, x_traj[:, 0], "k-", label="Ground Truth (q)")
    plt.plot(ts, x_traj[:, 1], "k--", label="Ground Truth (p)")
    initial_sol = symbolic_model.solve(ts, x_traj[0])
    plt.plot(ts, initial_sol[:, 0], "r-", label="Symbolic Model (q)")
    plt.plot(ts, initial_sol[:, 1], "r--", label="Symbolic Model (p)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"visualizations/symbolic_reconstruction_{physical_system.replace(' ', '_')}.png"
    )
    print(
        f"Plot saved as visualizations/symbolic_reconstruction_{physical_system.replace(' ', '_')}.png"
    )


if __name__ == "__main__":
    main()
