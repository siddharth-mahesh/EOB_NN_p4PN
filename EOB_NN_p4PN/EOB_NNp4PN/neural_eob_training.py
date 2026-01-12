from typing import Dict, Tuple, Union

import equinox as eqx
import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn import Neural_EOB

def loss(model, x_pred,y_pred):
    y_model = model(x_pred)
    return jnp.average(jax.vmap(physics_informed_loss)(y_pred,y_model))

def physics_informed_loss(y_pred,y_model):
    """
    Break down the loss into energy and dephasing loss.
    The output data is given as a set of (timesteps,[time(t),strain(h(t))]).
    The physics informed loss is given by a combination of
    1. Energy loss: P_GW = |\dot{h_pred}|^2 - |\dot{h_model}|^2
    2. Accumulated phase loss: \Delta \phi = \int_{t_0}^{t_f} (\phi_pred(t) - \phi_model(t)) dt
    3. Merger time loss: \Delta t = t_{merge, pred} - t_{merge, model}

    Args:
        y_pred (jnp.ndarray): The trusted output.
        y_model (jnp.ndarray): The neural network outputs.
    """
    t_merger_pred = jnp.real(y_pred[-1, 0])
    h_pred = y_pred[:, 1]
    hdot_pred = jnp.gradient(h_pred, y_pred[:,0])
    accumulated_phi_pred = jsp.integrate.trapezoid(jnp.unwrap(jnp.angle(h_pred)), y_pred[:,0])
    t_merger_model = jnp.real(y_model[-1, 0])
    h_model = y_model[:, 1]
    hdot_model = jnp.gradient(h_model, y_model[:,0])
    accumulated_phi_model = jsp.integrate.trapezoid(jnp.unwrap(jnp.angle(h_model)), y_model[:,0])
    return jnp.real(jnp.sum((jnp.abs(hdot_pred)**2 - jnp.abs(hdot_model)**2)**2 + (accumulated_phi_pred - accumulated_phi_model) ** 2 + (t_merger_pred - t_merger_model) ** 2))

def train_dhnn_model(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    val_data: Tuple[jnp.ndarray, jnp.ndarray],
    model_params: Dict[str, int],
    training_params: Dict[str, Union[int, float]],
) -> Neural_EOB:
    """Trains the Neural EOB model using the provided data and parameters.

    Args:
        train_data (Tuple[jnp.ndarray, jnp.ndarray]): The training data.
        val_data (Tuple[jnp.ndarray, jnp.ndarray]): The validation data.
        model_params (Dict[str, int]): The model parameters.
        training_params (Dict[str, int]): The training parameters.

    Returns:
        Neural_EOB: The trained model.
    """
    key = model_params["key"]
    model = Neural_EOB(**model_params)
    optimizer = optax.adam(learning_rate=training_params["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train, y_train = train_data
    x_val, y_val = val_data

    @eqx.filter_jit
    def step(
        model,
        opt_state,
        x,
        y,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(training_params["epochs"]):
        key, key_train, key_val = jax.random.split(key, 3)
        # select training_params['batch_size'] samples from x_train and x_val
        train_sample_idx = jax.random.choice(
            key_train,
            jnp.arange(0, x_train.shape[0], dtype=int),
            shape=(training_params["batch_size"],),
            replace=False,
        )
        val_sample_idx = jax.random.choice(
            key_val,
            jnp.arange(0, x_val.shape[0], dtype=int),
            shape=(training_params["batch_size"],),
            replace=False,
        )
        x_train_sample = jnp.take(x_train, train_sample_idx, axis=0)
        y_train_sample = jnp.take(y_train, train_sample_idx, axis=0)
        x_val_sample = jnp.take(x_val, val_sample_idx, axis=0)
        y_val_sample = jnp.take(y_val, val_sample_idx, axis=0)
        model, opt_state, loss_value = step(
            model, opt_state, x_train_sample, y_train_sample
        )
        val_loss_value = loss(model, x_val_sample, y_val_sample)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value}, Val Loss: {val_loss_value}")
        if loss_value < 1e-7:
            break
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # training parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    training_params = {
        "learning_rate": 1e-3,
        "epochs": 5000,
        "batch_size": 5,
    }
    # model parameters
    model_params = {
        "key": key,
    }
    # load training data
    x_train = jnp.load("x_sxs_1em4.npy")
    y_train = jnp.load("y_sxs_1em4.npy")
    x_val = jnp.load("x_sxs_1em3.npy")
    y_val = jnp.load("y_sxs_1em3.npy")
    trained_model = train_dhnn_model((x_train, y_train), (x_val, y_val), model_params, training_params)