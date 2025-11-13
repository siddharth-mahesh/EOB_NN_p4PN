import os
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from EOB_NN_p4PN.Schwarzschild.hnn_class import HNN_Model
from EOB_NN_p4PN.Schwarzschild.hnn_data_generation import HamiltonianDataGenerator
from tqdm import tqdm

def loss(model, x, y):
    y_pred = model(x)
    return jnp.average(((y_pred - y)/y) ** 2)


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


def train_model(
    train_data,
    val_data,
    seed: int = 23, # optimal seed from trial runs
) -> float:
    """Trains the HNN model using the provided data
    Args:
        train_data (Tuple[jnp.ndarray, jnp.ndarray]): The training data.
        val_data (Tuple[jnp.ndarray, jnp.ndarray]): The validation data.
        seed (int): The random seed for weight initialization.

    Returns:
        DHNN_Model: The trained model.
    """
    model_key = jax.random.PRNGKey(seed)
    model_params = {
        "key": model_key,
        "hidden_dim": 4,
    }
    model = HNN_Model(**model_params)
    training_params = {
        "seed": 24,
        "epochs": 50,
        "batch_size": train_data[0].shape[0]//50,
        "learning_rate": 1e-3,
    }
    optimizer = optax.adam(learning_rate=training_params["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train, y_train = train_data
    x_val, y_val = val_data
    key_train = jax.random.PRNGKey(training_params["seed"])
    for epoch in range(training_params["epochs"]):
        key_train, key_perm = jax.random.split(key_train, 2)
        perm = jax.random.permutation(key_perm, x_train.shape[0])
        batch_x = x_train[perm][: training_params["batch_size"]]
        batch_y = y_train[perm][: training_params["batch_size"]]
        model, opt_state, train_loss = step(model, opt_state, optimizer, batch_x, batch_y)
        val_loss_value = loss(model, x_val, y_val)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss_value}")  
        if jnp.isnan(train_loss) or jnp.isnan(val_loss_value):
            break
        
    return model, val_loss_value


# --- Main Execution ---
if __name__ == "__main__":
    generator = HamiltonianDataGenerator()
    nu_key, rs_key = jax.random.split(jax.random.PRNGKey(24), 2)
    total_batch_size = 50
    rs = jax.random.uniform(rs_key, (total_batch_size, 1), minval=7.0, maxval=35.0)
    pphis = jnp.sqrt(rs**2/(rs-3 - 3))
    z0s_train = jnp.hstack([rs,jnp.zeros((total_batch_size,1)),jnp.zeros((total_batch_size,1)),pphis])
    x , y = generator(z0s_train)
    val_data = (x[:total_batch_size//5], y[:total_batch_size//5])
    train_data = (x[total_batch_size//5:], y[total_batch_size//5:])
    model, val_loss_value = train_model(train_data, val_data)
    print(f"Val Loss: {val_loss_value}")
    