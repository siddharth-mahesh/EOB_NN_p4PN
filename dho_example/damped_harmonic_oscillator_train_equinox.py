import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
import optax
from typing import Dict, Tuple

from dho_dhnn_jax import DHNN_Model, MLP
from dho_data_jax import DampedHarmonicOscillator

def train_dhnn_model(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    val_data: Tuple[jnp.ndarray, jnp.ndarray],
    model_params: Dict[str, int],
    training_params: Dict[str, int]
    ) -> DHNN_Model:
    """Trains the DHNN model using the provided data and parameters.

    Args:
        train_data (Tuple[jnp.ndarray, jnp.ndarray]): The training data.
        val_data (Tuple[jnp.ndarray, jnp.ndarray]): The validation data.
        model_params (Dict[str, int]): The model parameters.
        training_params (Dict[str, int]): The training parameters.

    Returns:
        DHNN_Model: The trained model.
    """
    key = jax.random.PRNGKey(training_params['seed'])
    model = DHNN_Model(key=key, **model_params)
    optimizer = optax.adam(learning_rate=training_params['learning_rate'])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train , y_train = train_data
    x_val , y_val = val_data
    def loss(model, x, y):
        y_pred = model(x)
        return jnp.sum((y_pred - y)**2)
    
    @eqx.filter_jit
    def step(
        model,
        opt_state,
        x,
        y,
    ):
        loss_value , grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(training_params['epochs']):
        key , key_train , key_val = jax.random.split(key, 3)
        #select training_params['batch_size'] samples from x_train and x_val
        train_sample_idx = jax.random.choice(key_train, jnp.arange(0,x_train.shape[0], dtype=int), shape=(training_params['batch_size'],), replace=False)
        val_sample_idx = jax.random.choice(key_val, jnp.arange(0,x_val.shape[0], dtype=int), shape=(training_params['batch_size'],), replace=False)
        x_train_sample = jnp.take(x_train, train_sample_idx, axis=0)
        y_train_sample = jnp.take(y_train, train_sample_idx, axis=0)
        x_val_sample = jnp.take(x_val, val_sample_idx, axis=0)
        y_val_sample = jnp.take(y_val, val_sample_idx, axis=0)
        model, opt_state, loss_value = step(model, opt_state, x_train_sample, y_train_sample)
        val_loss_value = loss(model, x_val_sample, y_val_sample)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value}, Val Loss: {val_loss_value}')
        if loss_value < 1e-7:
            break
    return model

# --- Main Execution ---
if __name__ == '__main__':
    # training parameters
    training_params = {
        'seed': 42,
        'learning_rate': 1e-3,
        'epochs': 5000,
        'batch_size': 128,
    }
    # model parameters
    model_params = {
        'hidden_dim': 256,
    }
    # generate training data
    key = jax.random.PRNGKey(8765)
    key , key_prim , key_damp = jax.random.split(key,3)
    prims = jax.random.uniform(key_prim, (52, 2), minval=-1, maxval=1)
    dampings = 10**jax.random.uniform(key_damp, (52, 1), minval=jnp.log10(.1), maxval=jnp.log10(2))
    x0s_train = jnp.hstack([prims,dampings])
    data_class = DampedHarmonicOscillator()
    x_train , y_train = data_class(x0s_train, rhs=True)
    key ,key_prim, key_damp = jax.random.split(key,3)
    prims = jax.random.uniform(key_prim, (13, 2), minval=-1, maxval=1)
    dampings = 10**jax.random.uniform(key_damp, (13, 1), minval=jnp.log10(.1), maxval=jnp.log10(2))
    x0s_val = jnp.hstack([prims,dampings])
    x_val , y_val = data_class(x0s_val, rhs=True)
    trained_model = train_dhnn_model((x_train, y_train), (x_val, y_val), model_params, training_params)
    

