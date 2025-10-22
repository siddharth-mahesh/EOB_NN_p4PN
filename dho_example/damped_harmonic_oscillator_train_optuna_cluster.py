import os
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
    shared_project_dir = "/users/sm0193/scratch/EOB_NN_p4PN/dho_example"
    x_train = np.loadtxt("x_train.dat")
    y_train = np.loadtxt("y_train.dat")
    x_val = np.loadtxt("x_val.dat")
    y_val = np.loadtxt("y_val.dat")
    training_data = (x_train,y_train)
    validation_data = (x_val,y_val)
    db_name = "optuna_optimize_dho.db"
    db_path = os.path.join(shared_project_dir,db_name)
    db_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name="distributed_dho_hpo",
        storage=db_url,
        load_if_exists=True,
        direction="minimize"
    )
    study.optimize(
        lambda trial: objective(trial, training_data, validation_data), n_trials=2
    )
