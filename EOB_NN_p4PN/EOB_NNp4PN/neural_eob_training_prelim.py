from typing import Dict, Tuple, Union

import equinox as eqx
import jax
jax.config.update("jax_debug_nans", False)
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module_prelim import Neural_EOB

def vector_field_loss(model, x, y, weights=None):
    """
    Compute the distance between the predicted vector field and the true vector field.

    Args:
        model: The neural EOB model.
        x: The input data.
        y: The true vector field.
        weights: Optional array of weights to apply to the squared error of each component.

    Returns:
        The distance between the predicted vector field and the true vector field.
    """
    y_pred = model(x)
    squared_error = jnp.abs(y_pred - y)**2
    if weights is not None:
        squared_error = squared_error * weights
    return jnp.mean(squared_error)

def single_case_PN_rescaled_vector_field_loss(x, y, y_nn):
    u = 1.0 / x[1]
    # Keep the component dimension (shape: (4,)) so it can broadcast with weights (shape: (4,))
    squared_error = jnp.abs(y_nn - y)**2 / (u**8)
    return squared_error


def PN_rescaled_vector_field_loss(model, x, y, weights=None):
    """
    Compute the distance between the predicted vector field and the true vector field.

    Args:
        model: The neural EOB model.
        x: The input data.
        y: The true vector field.
        weights: Optional array of weights to apply to the squared error of each component.

    Returns:
        The distance between the predicted vector field and the true vector field.
    """
    y_pred = model(x)
    squared_error = jax.vmap(single_case_PN_rescaled_vector_field_loss,in_axes=(0,0,0))(x,y,y_pred)
    if weights is not None:
        squared_error = squared_error * weights
    return jnp.mean(squared_error)

def train_dhnn_model_prelim(
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
    
    total_steps = training_params["epochs"]
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-3,
        peak_value=5e-2,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-4
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(.5),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # Calculate inverse variance weights based on RESIDUALS
    model_zero = model
    model_zero = eqx.tree_at(lambda m: m.A_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.D_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.Q_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.f_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.delta_scale, model_zero, 0.0)
    
    y_base = model_zero(x_train)
    residual_sim = y_train - y_base
    u_var = 1.0 / x_train[:, 1]
    rescaled_residual = jnp.abs(residual_sim) / (u_var[:, None]**5)
    
    variances = jnp.var(rescaled_residual, axis=0)
    raw_weights = 1.0 / (variances + 1e-10)
    
    num_components = y_train.shape[1]
    weights = raw_weights / jnp.sum(raw_weights) * num_components
    print("Computed Loss Weights:", weights)

    @eqx.filter_jit
    def step(
        model,
        opt_state,
        x,
        y,
        weights
    ):
        loss_value, grads = eqx.filter_value_and_grad(PN_rescaled_vector_field_loss)(model, x, y, weights)
        grads = eqx.filter(grads, eqx.is_array)
        opt_state = eqx.filter(opt_state, eqx.is_array)
        model_ = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, model_)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(training_params["epochs"]):
        key, key_train, key_val = jax.random.split(key, 3)
        train_sample_idx = jax.random.choice(
            key_train,
            jnp.arange(0, x_train.shape[0], dtype=int),
            shape=(training_params["batch_size"],),
            replace=False,
        )
        x_train_sample = jnp.take(x_train, train_sample_idx, axis=0)
        y_train_sample = jnp.take(y_train, train_sample_idx, axis=0)
        model, opt_state, loss_value = step(
            model, opt_state, x_train_sample, y_train_sample, weights
        )
        val_loss_value = PN_rescaled_vector_field_loss(model, x_val, y_val, weights)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value}, Val Loss: {val_loss_value}")
        if val_loss_value < 1e-6:
            break
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # training parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    training_params = {
        "learning_rate": 1e-2,
        "epochs": 5000,
        "batch_size": 1000,
    }
    # model parameters
    model_params = {
        "key": key,
    }
    # load training data
    x_train = jnp.load("seob_x_train_prelim.npy")
    y_train = jnp.load("seob_y_train_prelim.npy")
    x_val = jnp.load("seob_x_val_prelim.npy")
    y_val = jnp.load("seob_y_val_prelim.npy")
    # train model
    trained_model = train_dhnn_model_prelim((x_train, y_train), (x_val, y_val), model_params, training_params)