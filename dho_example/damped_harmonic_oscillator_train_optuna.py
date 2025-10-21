from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import optuna
from dho_data_jax import DampedHarmonicOscillator
from dho_dhnn_jax import DHNN_Model


def loss(model, x, y):
    y_pred = model(x)
    return jnp.sum((y_pred - y) ** 2)


@eqx.filter_jit
def step(
    model,
    opt_state,
    optimizer,
    x,
    y,
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def objective(
    trial: optuna.Trial,
    train_data,
    val_data,
) -> float:
    """Trains the DHNN model using the provided data and parameters.

    Args:
        trial (optuna.Trial): The trial object.
        train_data (Tuple[jnp.ndarray, jnp.ndarray]): The training data.
        val_data (Tuple[jnp.ndarray, jnp.ndarray]): The validation data.

    Returns:
        DHNN_Model: The trained model.
    """
    model_seed = trial.suggest_int("model_seed", 0, 10000)
    model_key = jax.random.PRNGKey(model_seed)
    model_params = {
        "key": model_key,
        "hidden_dim": trial.suggest_int("hidden_dim", 1, 1024),
    }
    model = DHNN_Model(**model_params)
    training_params = {
        "seed": trial.suggest_int("training_seed", 0, 10000),
        "epochs": trial.suggest_int("epochs", 1, 10000),
        "batch_size": trial.suggest_int("batch_size", 1, 1024),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
    }
    optimizer = optax.adam(learning_rate=training_params["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train, y_train = train_data
    x_val, y_val = val_data
    key_train = jax.random.PRNGKey(training_params["seed"])
    for epoch in range(training_params["epochs"]):
        key_train, key_perm = jax.random.split(key_train, 2)
        # select training_params['batch_size'] samples from x_train and x_val
        perm = jax.random.permutation(key_perm, x_train.shape[0])

        # Simple batching for demonstration
        batch_x = x_train[perm][: training_params["batch_size"]]
        batch_y = y_train[perm][: training_params["batch_size"]]
        model, opt_state, _ = step(model, opt_state, optimizer, batch_x, batch_y)
    val_loss_value = loss(model, x_val, y_val)
    return val_loss_value


# --- Main Execution ---
if __name__ == "__main__":
    key = jax.random.PRNGKey(8765)
    key, key_prim, key_damp = jax.random.split(key, 3)
    prims = jax.random.uniform(key_prim, (200, 2), minval=-1, maxval=1)
    dampings = 10 ** jax.random.uniform(
        key_damp, (200, 1), minval=jnp.log10(0.1), maxval=jnp.log10(2)
    )
    x0s_train = jnp.hstack([prims, dampings])
    data_class = DampedHarmonicOscillator()
    training_data = data_class(x0s_train, rhs=True)
    key, key_prim, key_damp = jax.random.split(key, 3)
    prims = jax.random.uniform(key_prim, (50, 2), minval=-1, maxval=1)
    dampings = 10 ** jax.random.uniform(
        key_damp, (50, 1), minval=jnp.log10(0.1), maxval=jnp.log10(2)
    )
    x0s_val = jnp.hstack([prims, dampings])
    validation_data = data_class(x0s_val, rhs=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, training_data, validation_data), n_trials=2
    )
