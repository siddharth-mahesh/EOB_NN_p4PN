from typing import Dict, Tuple, Union

import equinox as eqx
import jax
jax.config.update("jax_debug_nans", False)
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module import Neural_EOB

def single_strain_loss(y_true,y_pred):
    t_pred = jnp.real(y_pred[:, 0]) - jnp.real(y_pred[-1, 0])
    h_pred = y_pred[:,1]
    h_pred_safe = h_pred + 1e-15j
    h_true_safe = y_true[:,1] + 1e-15j
    a_pred = jnp.abs(h_pred_safe)
    phi_pred = jnp.unwrap(jnp.angle(h_pred_safe)) - jnp.unwrap(jnp.angle(h_pred_safe))[-1]
    t_true = jnp.real(y_true[:,0]) - jnp.real(y_true[-1,0])
    a_true = jnp.abs(h_true_safe)   
    phi_true = jnp.unwrap(jnp.angle(h_true_safe)) - jnp.unwrap(jnp.angle(h_true_safe))[-1]
    a_overlap = jnp.interp(t_pred, t_true, a_true, left=a_true[0], right=a_true[-1])
    phi_overlap = jnp.interp(t_pred, t_true, phi_true, left=phi_true[0], right=phi_true[-1])
    return jnp.mean(jnp.abs(a_pred - a_overlap)**2 + jnp.abs(phi_pred - phi_overlap)**2)

def single_strain_mismatch_loss(y_true,y_pred):
    t_pred = jnp.real(y_pred[:, 0]) - jnp.real(y_pred[-1, 0])
    h_pred = y_pred[:, 1]
    
    t_true = jnp.real(y_true[:, 0]) - jnp.real(y_true[-1, 0])
    h_true = y_true[:, 1]
    
    # Interpolate true strain to predicted time points
    h_true_interp_real = jnp.interp(t_pred, t_true, jnp.real(h_true), left=jnp.real(h_true[0]), right=jnp.real(h_true[-1]))
    h_true_interp_imag = jnp.interp(t_pred, t_true, jnp.imag(h_true), left=jnp.imag(h_true[0]), right=jnp.imag(h_true[-1]))
    h_true_interp = h_true_interp_real + 1j * h_true_interp_imag
    
    # Compute normalized mismatch: 1 - |<h_pred | h_true>|^2 / (<h_pred|h_pred> <h_true|h_true>)
    inner_hh = jnp.sum(h_pred * jnp.conj(h_true_interp))
    inner_pp = jnp.sum(h_pred * jnp.conj(h_pred))
    inner_tt = jnp.sum(h_true_interp * jnp.conj(h_true_interp))
    
    mismatch = 1.0 - (jnp.abs(inner_hh)**2) / (jnp.real(inner_pp) * jnp.real(inner_tt) + 1e-15)
    return mismatch

def direct_waveform_loss(model, x , y):
    y_pred = model(x)
    batched_loss = jax.vmap(single_strain_mismatch_loss, in_axes=(0,0))(y,y_pred)
    return jnp.mean(batched_loss)
    
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
    
    total_steps = training_params["epochs"]
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=training_params["learning_rate"],
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-4
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    #optimizer = optax.adam(learning_rate=training_params["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    x_train, y_train = train_data
    x_val, y_val = val_data

    @eqx.filter_jit
    def step(
        model,
        opt_state,
        x,
        y
    ):
        loss_value, grads = eqx.filter_value_and_grad(direct_waveform_loss)(model, x, y)
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
        x_train_sample = jnp.take(x_train, train_sample_idx, axis=0)
        y_train_sample = jnp.take(y_train, train_sample_idx, axis=0)
        model, opt_state, loss_value = step(
            model, opt_state, x_train_sample, y_train_sample
        )
        val_loss_value = direct_waveform_loss(model, x_val, y_val)
        print(f"Epoch {epoch}, Loss: {loss_value}, Val Loss: {val_loss_value}")
        if loss_value < 1e-3:
            break

    return model


# --- Main Execution ---
if __name__ == "__main__":
    # training parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    training_params = {
        "learning_rate": 1e-2,
        "epochs": 500,
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