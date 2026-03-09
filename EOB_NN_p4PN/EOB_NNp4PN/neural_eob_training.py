"""
Main training script for the Neural EOB model.

Defines the core loss functions (direct waveform loss, merger time loss, and 
physics-informed loss components) and provides the `train_dhnn_model` 
routine to optimize the Neural EOB components using JAX and Optax.
"""

from typing import Dict, Tuple, Union

import equinox as eqx
import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module import Neural_EOB

def loss_stage_zero(model, x_pred,y_pred):
    y_model = model(x_pred)
    batched_loss = jax.vmap(direct_waveform_loss, in_axes=(0,0))(y_pred,y_model)
    return jnp.average(batched_loss)

def loss_first_stage(model, x_pred,y_pred):
    y_model = model(x_pred)
    batched_loss = jax.vmap(physics_informed_loss_preliminary, in_axes=(0,0))(y_pred,y_model)
    return jnp.average(batched_loss)

def loss_second_stage(model, x_pred,y_pred):
    y_model = model(x_pred)
    batched_loss = jax.vmap(physics_informed_loss, in_axes=(0,0))(y_pred,y_model)
    return jnp.average(batched_loss)

def direct_waveform_loss(y_pred,y_model):
    t_end = jnp.min(jnp.array([jnp.real(y_pred[-1,0]),jnp.real(y_model[-1,0])]))
    t_eval = jnp.linspace(0, t_end, 256)
    h_pred = jnp.interp(t_eval, jnp.real(y_pred[:, 0]), y_pred[:, 1])
    h_model = jnp.interp(t_eval, jnp.real(y_model[:, 0]), y_model[:, 1])
    return jnp.mean(jnp.abs(h_pred - h_model)**2)

def merger_time_loss(y_pred,y_model):
    """
    Compute the merger time loss, defined by
    \Delta t = t_{merge, pred} - t_{merge, model}
    
    Args:
        y_pred (jnp.ndarray): The trusted output.
        y_model (jnp.ndarray): The neural network outputs.
    """
    t_merger_pred = jnp.real(y_pred[-1, 0])
    t_merger_model = jnp.real(y_model[-1, 0])
    # threshold merger time by allowable error margin
    return ((t_merger_pred - t_merger_model)/5.) ** 2

def physics_informed_loss(y_pred,y_model):
    """
    Break down the loss into energy and dephasing loss for final training(up to merger).
    The output data is given as a set of (timesteps,[time(t),strain(h(t))]).
    The physics informed loss is given by a combination of
    1. Energy loss: \Delta E_GW = \int_{t=0}^{t_{merge}} (|\dot{h_pred}(t)|^2 - |\dot{h_model}(t)|^2) dt
    2. Accumulated phase loss: \Delta \phi = \int_{t=0}^{t_{merge}} (\phi_pred(t) - \phi_model(t))^2 dt

    Args:
        y_pred (jnp.ndarray): The trusted output.
        y_model (jnp.ndarray): The neural network outputs.
    """
    t_end = jnp.min(jnp.array([jnp.real(y_pred[-1,0]),jnp.real(y_model[-1,0])]))
    t_eval = jnp.linspace(0, t_end, 1024)
    h_pred = jnp.interp(t_eval, jnp.real(y_pred[:, 0]), y_pred[:, 1])
    P_GW_pred = jnp.abs(jnp.gradient(h_pred, t_eval))**2
    h_model = jnp.interp(t_eval, jnp.real(y_model[:, 0]), y_model[:, 1])
    P_GW_model = jnp.abs(jnp.gradient(h_model, t_eval))**2
    energy_squared_loss = jsp.integrate.trapezoid((P_GW_model - P_GW_pred)**2, t_eval)
    accumulated_phase_squared_loss = jsp.integrate.trapezoid((jnp.unwrap(jnp.angle(h_pred)) - jnp.unwrap(jnp.angle(h_model)))**2, t_eval)
    return accumulated_phase_squared_loss

def physics_informed_loss_preliminary(y_pred,y_model):
    """
    Break down the loss into energy and dephasing loss for preliminary training(up to 500M).
    The output data is given as a set of (timesteps,[time(t),strain(h(t))]).
    The physics informed loss is given by a combination of
    1. Energy loss: \Delta E_GW = \int_{t=0}^{t=500M} (|\dot{h_pred}(t)|^2 - |\dot{h_model}(t)|^2) dt
    2. Accumulated phase loss: \Delta \phi = \int_{t=0}^{t=500M} (\phi_pred(t) - \phi_model(t))^2 dt

    Args:
        y_pred (jnp.ndarray): The trusted output.
        y_model (jnp.ndarray): The neural network outputs.
    """
    t_eval = jnp.linspace(0, 500, 1024)
    h_pred = jnp.interp(t_eval, jnp.real(y_pred[:, 0]), y_pred[:, 1])
    P_GW_pred = jnp.abs(jnp.gradient(h_pred, t_eval))**2
    h_model = jnp.interp(t_eval, jnp.real(y_model[:, 0]), y_model[:, 1])
    P_GW_model = jnp.abs(jnp.gradient(h_model, t_eval))**2
    energy_squared_loss = jsp.integrate.trapezoid((P_GW_model - P_GW_pred)**2, t_eval)
    accumulated_phase_squared_loss = jsp.integrate.trapezoid(
        (
            jnp.unwrap(jnp.angle(h_pred)) 
            - jnp.unwrap(jnp.angle(h_model))
        )**2,
        t_eval
    )
    return accumulated_phase_squared_loss #+ energy_squared_loss

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
    optimizer = optax.optimistic_adam_v2(learning_rate=training_params["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train, y_train = train_data
    x_val, y_val = val_data

    @eqx.filter_jit
    def step_stage_zero(
        model,
        opt_state,
        x,
        y
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_stage_zero)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
        

    @eqx.filter_jit
    def step_first_stage(
        model,
        opt_state,
        x,
        y,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_first_stage)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(training_params["stage_zero_epochs"]):
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
        model, opt_state, loss_value = step_stage_zero(
            model, opt_state, x_train_sample, y_train_sample
        )
        val_loss_value = loss_stage_zero(model, x_val_sample, y_val_sample)
        print(f"Epoch {epoch}, Loss: {loss_value}, Val Loss: {val_loss_value}")
        if loss_value < 1e-6:
            break

    return model


# --- Main Execution ---
if __name__ == "__main__":
    # training parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    training_params = {
        "learning_rate": 1e-4,
        "stage_zero_epochs": 500,
        "stage_one_epochs": 500,
        "stage_two_epochs": 500,
        "batch_size": 10,
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
    # train model
    trained_model = train_dhnn_model((x_train, y_train), (x_val, y_val), model_params, training_params)