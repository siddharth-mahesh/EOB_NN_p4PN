"""Preliminary training loop for Neural-EOB RHS regression.

This module trains a differentiable EOB perturber to reproduce the SEOBNRv5
equations-of-motion right-hand side (RHS) on pre-generated samples.

Data conventions used throughout this file:
- `x` has shape `(N, 5)` and is ordered as `[nu, r, phi, p_rstar, p_phi]`.
- `y` has shape `(N, 4)` and is the target RHS
  `[dr/dt, dphi/dt, dp_rstar/dt, dp_phi/dt]`.

Training strategy:
1. Construct a strict 3PN baseline by zeroing neural correction scales.
2. Train with a staged objective:
   - escape phase: amplified residual learning + high-u curriculum
   - early phase: standardized vector-field fitting
   - blend phase: interpolate toward a residual-to-baseline objective
   - optional late phase: add Jacobian smoothness regularization
3. Use warmup + cosine-decay Adam with gradient clipping.

The implementation is intentionally explicit (instead of heavily abstracted) so
loss diagnostics, schedules, and numerical guards can be inspected and edited
quickly during convergence experiments.

Experiment modes:
- `eob_v2`: structured EOB-potential model with baseline/residual curriculum.
- `blackbox`: generic dissipative Hamiltonian NN baseline for trainability
  diagnostics with minimal EOB structure assumptions.
- `hybrid_eob`: EOB-form Hamiltonian/flux with fully learned A/D/Q/f potentials.
"""

from typing import Dict, Tuple, Union
from pathlib import Path

import equinox as eqx
import jax
jax.config.update("jax_debug_nans", False)
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module_prelim import Neural_EOB
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module_prelim_v2 import Neural_EOB_V2
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module_blackbox import BlackBoxDHNN
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module_hybrid import Hybrid_EOB_DHNN

def vector_field_loss(model, x, y, weights=None):
    """Plain MSE on RHS components.

    This helper is kept as a minimal reference loss and for quick debugging.
    The main training path below uses standardized losses.
    """
    y_pred = model(x)
    squared_error = jnp.abs(y_pred - y)**2
    if weights is not None:
        squared_error = squared_error * weights
    return jnp.mean(squared_error)

def single_case_PN_rescaled_vector_field_loss(x, y, y_nn):
    """Single-sample PN-rescaled squared error.

    Error is divided by `u^8` (`u = 1/r`) to upweight weak-field mismatch where
    high-PN effects are relatively suppressed.
    """
    r_safe = jnp.maximum(jnp.abs(x[1]), 1e-12)
    u = 1.0 / r_safe
    # Keep the component dimension (shape: (4,)) so it can broadcast with weights (shape: (4,))
    squared_error = jnp.abs(y_nn - y)**2 / (u**8)
    return squared_error


def PN_rescaled_vector_field_loss(model, x, y, weights=None):
    """Batch PN-rescaled loss with finite-output guard.

    If any model output is non-finite, returns a large fallback value to prevent
    invalid gradients from corrupting optimizer state.
    """
    y_pred = model(x)
    finite = jnp.all(jnp.isfinite(y_pred))

    def finite_loss(y_pred_local):
        squared_error = jax.vmap(single_case_PN_rescaled_vector_field_loss, in_axes=(0, 0, 0))(x, y, y_pred_local)
        if weights is not None:
            squared_error = squared_error * weights
        return jnp.mean(squared_error)

    def fallback_loss(_):
        return jnp.array(1e6, dtype=x.dtype)

    return jax.lax.cond(finite, finite_loss, fallback_loss, y_pred)


def _component_scales(y, eps=1e-8):
    """Per-output standard deviations used to standardize losses."""
    return jnp.maximum(jnp.std(y, axis=0), eps)


def _component_rms(y, eps=1e-8):
    """Per-output RMS magnitude used for zero-centered residual normalization."""
    return jnp.maximum(jnp.sqrt(jnp.mean(y**2, axis=0)), eps)


def _assign_u_bins(x, edges, eps=1e-12):
    """Assign each sample to a compactness bin, with compactness `u = 1/r`."""
    r_safe = jnp.maximum(jnp.abs(x[:, 1]), eps)
    u = 1.0 / r_safe
    bin_idx = jnp.searchsorted(edges[1:], u, side="right")
    return jnp.clip(bin_idx, 0, edges.shape[0] - 2)


def _build_u_binned_residual_scales(
    x,
    delta,
    num_bins=8,
    eps=1e-8,
    min_bin_count=32,
    floor_frac=0.1,
):
    """Build per-`u`-bin residual scales for heteroscedastic normalization.

    Residuals are defined relative to the baseline (`delta = y_true - y_base`).
    Each bin gets per-component scales from residual RMS (around zero), with guards:
    - sparse bins fallback to global scales (`min_bin_count`)
    - scales are floored by `floor_frac * global_scale` to avoid tiny divisors
    """
    r_safe = jnp.maximum(jnp.abs(x[:, 1]), eps)
    u = 1.0 / r_safe
    u_min = float(jnp.min(u))
    u_max = float(jnp.max(u))
    if (num_bins <= 1) or (u_max <= u_min):
        edges = jnp.array([u_min, u_max + eps], dtype=u.dtype)
        return edges, jnp.zeros((x.shape[0],), dtype=jnp.int32), _component_rms(delta, eps)[None, :]

    edges = jnp.linspace(u_min, u_max, num_bins + 1)
    bin_idx = _assign_u_bins(x, edges, eps)
    global_scales = _component_rms(delta, eps)
    floor_frac = max(0.0, float(floor_frac))
    scale_floor = floor_frac * global_scales
    scales = []
    for b in range(num_bins):
        mask = (bin_idx == b).astype(delta.dtype)[:, None]
        count = jnp.sum(mask[:, 0])
        denom = jnp.maximum(count, 1.0)
        rms_b = jnp.sqrt(jnp.maximum(jnp.sum((delta**2) * mask, axis=0) / denom, eps**2))
        rms_b = jnp.where(count >= min_bin_count, rms_b, global_scales)
        rms_b = jnp.maximum(rms_b, scale_floor)
        scales.append(rms_b)
    scale_table = jnp.stack(scales, axis=0)
    return edges, bin_idx, scale_table


def _standardized_vf_loss_from_pred(y_pred, y_true, vf_scales, eps=1e-8):
    """Standardized vector-field MSE."""
    err = (y_pred - y_true) / (vf_scales[None, :] + eps)
    return jnp.mean(err**2)


def _componentwise_relative_error_metrics_from_pred(y_pred, y_true, eps=1e-12):
    """Componentwise relative error metrics.

    Relative error per component is defined as:
        abs((y_pred_i - y_true_i) / y_true_i)
    with denominator regularized by `eps` for numerical stability.
    """
    rel_abs = jnp.abs(y_pred - y_true) / (jnp.abs(y_true) + eps)
    rel_abs_mean = jnp.mean(rel_abs)
    rel_abs_comp = jnp.mean(rel_abs, axis=0)
    return rel_abs_mean, rel_abs_comp


def _relative_error_matrix_from_pred(y_pred, y_true, eps=1e-12):
    """Per-sample, per-component relative error matrix."""
    return jnp.abs((y_pred - y_true) / (jnp.abs(y_true) + eps))


def _relative_error_summary(rel: Union[jnp.ndarray, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
    """Robust summary for relative-error arrays with finite filtering."""
    arr = np.asarray(rel, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected relative error array with shape (N, C), got {arr.shape}.")
    arr = np.abs(arr)
    n_comp = arr.shape[1]

    finite_flat = arr[np.isfinite(arr)]
    if finite_flat.size == 0:
        return {
            "mean": np.inf,
            "p95": np.inf,
            "comp_mean": np.full((n_comp,), np.inf, dtype=np.float64),
            "comp_p95": np.full((n_comp,), np.inf, dtype=np.float64),
            "finite_ratio": 0.0,
        }

    comp_mean = np.zeros((n_comp,), dtype=np.float64)
    comp_p95 = np.zeros((n_comp,), dtype=np.float64)
    for j in range(n_comp):
        col = arr[:, j]
        col_f = col[np.isfinite(col)]
        if col_f.size == 0:
            comp_mean[j] = np.inf
            comp_p95[j] = np.inf
        else:
            comp_mean[j] = float(np.mean(col_f))
            comp_p95[j] = float(np.quantile(col_f, 0.95))

    return {
        "mean": float(np.mean(finite_flat)),
        "p95": float(np.quantile(finite_flat, 0.95)),
        "comp_mean": comp_mean,
        "comp_p95": comp_p95,
        "finite_ratio": float(finite_flat.size / arr.size),
    }


def _split_supervision_data(data, split_name: str):
    """Accept (x, y) or (x, y, e_rel_ref)."""
    if not isinstance(data, (tuple, list)):
        raise TypeError(f"{split_name} data must be tuple/list, got {type(data)}.")
    if len(data) < 2:
        raise ValueError(f"{split_name} data must have at least (x, y).")
    x, y = data[0], data[1]
    e_rel_ref = data[2] if len(data) >= 3 else None
    return x, y, e_rel_ref


def save_model_weights(model, path: str):
    """Save model parameters for future reloading."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(out_path), model)
    print(f"Saved model weights to: {out_path}")


def load_model_weights(model, path: str, strict: bool = True):
    """Load model parameters into an existing model structure."""
    in_path = Path(path)
    if not in_path.exists():
        msg = f"Weight file not found: {in_path}"
        if strict:
            raise FileNotFoundError(msg)
        print("WARNING:", msg)
        return model
    model_loaded = eqx.tree_deserialise_leaves(str(in_path), model)
    print(f"Loaded model weights from: {in_path}")
    return model_loaded


def _standardized_residual_loss_from_pred(y_pred, y_true, y_base, residual_scales, eps=1e-8):
    """Standardized residual-to-baseline MSE."""
    delta_pred = y_pred - y_base
    delta_true = y_true - y_base
    err = (delta_pred - delta_true) / (residual_scales + eps)
    return jnp.mean(err**2)


def _standardized_residual_loss_from_delta(delta_pred, delta_true, residual_scales, eps=1e-8):
    """Standardized residual MSE from pre-computed residuals."""
    err = (delta_pred - delta_true) / (residual_scales + eps)
    return jnp.mean(err**2)


def _validate_rhs_targets(y: jnp.ndarray, split_name: str, strict: bool = True, tol: float = 1e-14):
    """Detect obvious target-layout issues before expensive training starts."""
    if y.ndim != 2 or y.shape[1] != 4:
        msg = f"{split_name}: expected RHS shape (N, 4), got {tuple(y.shape)}."
        raise ValueError(msg)

    duplicate_pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            max_abs_diff = jnp.max(jnp.abs(y[:, i] - y[:, j]))
            if float(max_abs_diff) <= tol:
                duplicate_pairs.append((i, j))

    omega_nonpos_ratio = float(jnp.mean(y[:, 1] <= 0.0))
    issues = []
    if duplicate_pairs:
        issues.append(f"duplicate RHS columns detected: {duplicate_pairs}")
    if omega_nonpos_ratio > 0.95:
        issues.append(
            "column 1 is almost entirely non-positive "
            f"(non-positive ratio={omega_nonpos_ratio:.3f}); "
            "expected dphi/dt is usually positive in this trainer's convention"
        )

    if not issues:
        return

    msg = (
        f"{split_name}: potential target-layout anomaly. "
        + "; ".join(issues)
        + ". If your dataset uses a different component ordering/sign convention, "
        "disable strict mode with training_params['strict_target_validation']=False "
        "and map targets before training."
    )
    if strict:
        raise ValueError(msg)
    print("WARNING:", msg)


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean over masked samples (returns 0 when mask is empty)."""
    mask_f = mask.astype(values.dtype)
    denom = jnp.maximum(jnp.sum(mask_f), 1.0)
    return jnp.sum(values * mask_f) / denom


def _safe_log_abs(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Stable log(|x|)."""
    return jnp.log(jnp.maximum(jnp.abs(x), eps))

def train_dhnn_model_prelim(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    val_data: Tuple[jnp.ndarray, jnp.ndarray],
    model_params: Dict[str, int],
    training_params: Dict[str, Union[int, float]],
) -> Neural_EOB:
    """Train a preliminary Neural-EOB model on RHS supervision.

    Args:
        train_data:
            `(x_train, y_train)` where
            - `x_train`: shape `(N_train, 5)` in `[nu, r, phi, p_rstar, p_phi]`
            - `y_train`: shape `(N_train, 4)` RHS targets.
        val_data:
            `(x_val, y_val)` with the same conventions as train data.
        model_params:
            Constructor kwargs for the model. Optional key:
            - `"model_class"`: model type (defaults to `Neural_EOB`).
        training_params:
            Hyperparameters for optimizer and staged loss schedule. Relevant keys:
            - optimizer: `learning_rate`, `lr_init`, `lr_end`, `batch_size`,
              `adam_epochs`
            - staged objective: `stage0_epochs`, `blend_end_epochs`
              and `beta_start`
            - jacobian regularizer: `jacobian_start_epoch`,
              `jacobian_ramp_epochs`, `jacobian_weight`, `jacobian_batch_size`
            - residual scaling: `num_u_bins`, `min_bin_count`,
              `residual_scale_floor_frac`, `loss_eps`
            - escape phase: `escape_epochs`, `escape_gamma_start`,
              `escape_gamma_end`, `escape_weight`, `escape_margin`,
              `escape_u_threshold`, `escape_high_u_frac`, `escape_top_bins`
            - data safety: `strict_target_validation`

    Returns:
        Trained model instance (`model_class` from `model_params`).
    """
    model_class = model_params.get("model_class", Neural_EOB)
    model_kwargs = {k: v for k, v in model_params.items() if k != "model_class"}
    key = model_kwargs["key"]
    model = model_class(**model_kwargs)
    x_train, y_train, _ = _split_supervision_data(train_data, "train")
    x_val, y_val, _ = _split_supervision_data(val_data, "val")
    batch_size = int(training_params["batch_size"])
    strict_target_validation = bool(training_params.get("strict_target_validation", True))

    _validate_rhs_targets(y_train, "train", strict=strict_target_validation)
    _validate_rhs_targets(y_val, "val", strict=strict_target_validation)

    num_train_samples = x_train.shape[0]
    effective_batch_size = min(batch_size, num_train_samples)
    num_train_batches = max(1, num_train_samples // effective_batch_size)
    used_train_samples = num_train_batches * effective_batch_size
    dropped_train_samples = num_train_samples - used_train_samples

    lr_peak = float(training_params.get("learning_rate", 3e-4))
    lr_init = float(training_params.get("lr_init", 0.1 * lr_peak))
    lr_end = float(training_params.get("lr_end", 0.01 * lr_peak))

    total_steps = int(training_params["adam_epochs"]) * num_train_batches
    warmup_steps_requested = int(training_params.get("warmup_steps", int(0.1 * total_steps)))
    warmup_steps = min(max(0, warmup_steps_requested), max(0, total_steps - 1))
    decay_steps = max(total_steps, warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_init,
        peak_value=lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_end,
    )
    print(
        "LR schedule:",
        {
            "init": lr_init,
            "peak": lr_peak,
            "end": lr_end,
            "warmup_steps": warmup_steps,
            "warmup_steps_requested": warmup_steps_requested,
            "total_steps": total_steps,
            "decay_steps": decay_steps,
        },
    )

    stage0_epochs = int(training_params.get("stage0_epochs", 80))
    blend_end_epochs = int(training_params.get("blend_end_epochs", 600))
    beta_start = float(training_params.get("beta_start", 0.3))
    beta_start = min(max(beta_start, 0.0), 1.0)
    jacobian_start_epoch = int(training_params.get("jacobian_start_epoch", blend_end_epochs))
    jacobian_ramp_epochs = max(1, int(training_params.get("jacobian_ramp_epochs", 300)))
    jacobian_weight = float(training_params.get("jacobian_weight", 1e-5))
    jacobian_batch_size = max(
        1, min(int(training_params.get("jacobian_batch_size", 16)), effective_batch_size)
    )
    num_u_bins = int(training_params.get("num_u_bins", 8))
    min_bin_count = int(training_params.get("min_bin_count", 32))
    residual_scale_floor_frac = float(training_params.get("residual_scale_floor_frac", 0.1))
    escape_epochs = max(0, int(training_params.get("escape_epochs", 300)))
    escape_gamma_start = float(training_params.get("escape_gamma_start", 1.2))
    escape_gamma_end = float(training_params.get("escape_gamma_end", 1.0))
    escape_gamma_start = max(1.0, escape_gamma_start)
    escape_gamma_end = max(1.0, escape_gamma_end)
    escape_weight = float(training_params.get("escape_weight", 0.05))
    escape_margin = float(training_params.get("escape_margin", 0.1))
    escape_u_threshold = float(training_params.get("escape_u_threshold", 0.12))
    escape_high_u_frac = float(training_params.get("escape_high_u_frac", 0.1))
    escape_high_u_frac = min(max(escape_high_u_frac, 0.0), 1.0)
    escape_top_bins = max(1, int(training_params.get("escape_top_bins", 1)))
    eps = float(training_params.get("loss_eps", 1e-8))
    print(
        "Loss schedule:",
        {
            "stage0_epochs": stage0_epochs,
            "blend_end_epochs": blend_end_epochs,
            "beta_start": beta_start,
            "jacobian_start_epoch": jacobian_start_epoch,
            "jacobian_ramp_epochs": jacobian_ramp_epochs,
            "jacobian_weight": jacobian_weight,
            "jacobian_batch_size": jacobian_batch_size,
            "num_u_bins": num_u_bins,
            "min_bin_count": min_bin_count,
            "residual_scale_floor_frac": residual_scale_floor_frac,
            "escape_epochs": escape_epochs,
            "escape_gamma_start": escape_gamma_start,
            "escape_gamma_end": escape_gamma_end,
            "escape_weight": escape_weight,
            "escape_margin": escape_margin,
            "escape_u_threshold": escape_u_threshold,
            "escape_high_u_frac": escape_high_u_frac,
            "escape_top_bins": escape_top_bins,
        },
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Build a strict analytic 3PN baseline by disabling all learned corrections.
    # Residual losses are measured relative to this baseline.
    model_zero = model
    model_zero = eqx.tree_at(lambda m: m.A_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.D_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.Q_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.f_scale, model_zero, 0.0)
    model_zero = eqx.tree_at(lambda m: m.delta_scale, model_zero, 0.0)
    y_base_train = model_zero(x_train)
    y_base_val = model_zero(x_val)

    omega_base = y_base_train[:, 1]
    omega_nonpos_ratio = jnp.mean(omega_base <= 0.0)
    omega_min = jnp.min(omega_base)
    print(f"Baseline omega_min: {omega_min}, omega_nonpos_ratio: {omega_nonpos_ratio}")
    finite_ratio = jnp.mean(jnp.isfinite(y_base_train))
    print(f"Baseline finite ratio in y_base: {finite_ratio}")

    vf_scales = _component_scales(y_train, eps)
    delta_train = y_train - y_base_train
    u_edges, train_bin_idx, residual_scale_table = _build_u_binned_residual_scales(
        x_train,
        delta_train,
        num_bins=num_u_bins,
        eps=eps,
        min_bin_count=min_bin_count,
        floor_frac=residual_scale_floor_frac,
    )
    val_bin_idx = _assign_u_bins(x_val, u_edges, eps)
    val_residual_scales = jnp.take(residual_scale_table, val_bin_idx, axis=0)
    train_residual_scales = jnp.take(residual_scale_table, train_bin_idx, axis=0)
    vf_ref = jnp.maximum(
        _standardized_vf_loss_from_pred(y_base_train, y_train, vf_scales, eps), eps
    )
    res_ref = jnp.maximum(
        _standardized_residual_loss_from_pred(
            y_base_train, y_train, y_base_train, train_residual_scales, eps
        ),
        eps,
    )
    print("Vector-field component scales:", vf_scales)
    print(
        "Residual scale table stats:",
        {
            "shape": residual_scale_table.shape,
            "min": jnp.min(residual_scale_table),
            "max": jnp.max(residual_scale_table),
        },
    )
    print("Loss anchors:", {"vf_ref": vf_ref, "res_ref": res_ref})

    all_pool_idx = jnp.arange(num_train_samples, dtype=jnp.int32)
    high_bin_threshold = max(0, num_u_bins - escape_top_bins)
    train_bin_idx_np = np.asarray(jax.device_get(train_bin_idx))
    all_idx_np = np.arange(num_train_samples, dtype=np.int32)
    high_pool_np = all_idx_np[train_bin_idx_np >= high_bin_threshold]
    low_pool_np = all_idx_np[train_bin_idx_np < high_bin_threshold]
    high_pool_idx = jnp.asarray(high_pool_np, dtype=jnp.int32)
    low_pool_idx = jnp.asarray(low_pool_np, dtype=jnp.int32)
    if escape_epochs > 0:
        print(
            "Escape sampling pools:",
            {
                "high_pool_size": int(high_pool_idx.shape[0]),
                "low_pool_size": int(low_pool_idx.shape[0]),
                "high_bin_threshold": high_bin_threshold,
            },
        )

    if dropped_train_samples > 0:
        print(
            f"Dropping {dropped_train_samples} train samples each epoch "
            f"to keep a fixed JIT batch shape."
        )

    @eqx.filter_jit
    def step(
        model,
        opt_state,
        x,
        y,
        y_base,
        residual_scales,
        x_jac,
        vf_scales,
        vf_ref,
        res_ref,
        beta,
        jac_weight_now,
        escape_gamma,
        escape_weight_now,
        escape_u_threshold,
        escape_margin,
    ):
        """One optimizer step on a fixed-size batch.

        Main loss is a convex blend of:
        - standardized vector-field loss (`l_vf`)
        - standardized residual loss (`l_res`)

        Optional Jacobian penalty (`l_jac`) is activated/ramped by schedule to
        discourage pathological local stiffness while preserving early fitting
        freedom.
        """
        def loss_fn(m):
            y_pred = m(x)
            l_vf = _standardized_vf_loss_from_pred(y_pred, y, vf_scales, eps)
            delta_pred = y_pred - y_base
            delta_true = y - y_base
            l_res = _standardized_residual_loss_from_delta(
                delta_pred, delta_true, residual_scales, eps
            )
            l_vf_norm = l_vf / vf_ref
            l_res_norm = l_res / res_ref
            # Emphasize residual fitting during escape without changing the minimizer.
            l_res_obj = escape_gamma * l_res_norm
            main_loss = (1.0 - beta) * l_vf_norm + beta * l_res_obj

            def jacobian_term(_):
                def single_out(x_single):
                    return m(jnp.expand_dims(x_single, axis=0))[0]

                jac = jax.vmap(jax.jacfwd(single_out))(x_jac)
                return jnp.mean(jnp.square(jac))

            l_jac = jax.lax.cond(
                jac_weight_now > 0.0,
                jacobian_term,
                lambda _: jnp.array(0.0, dtype=x.dtype),
                operand=None,
            )

            def escape_term(_):
                r_safe = jnp.maximum(jnp.abs(x[:, 1]), eps)
                u = 1.0 / r_safe
                hi_mask = (u >= escape_u_threshold).astype(x.dtype)
                hi_count = jnp.sum(hi_mask)

                def with_hi(_):
                    sample_norm = jnp.sqrt(
                        jnp.sum(jnp.square(delta_pred / (residual_scales + eps)), axis=1) + eps
                    )
                    mean_norm = jnp.sum(sample_norm * hi_mask) / hi_count
                    return jax.nn.softplus(escape_margin - mean_norm)

                return jax.lax.cond(
                    hi_count > 0.0,
                    with_hi,
                    lambda __: jnp.array(0.0, dtype=x.dtype),
                    operand=None,
                )

            l_escape = jax.lax.cond(
                escape_weight_now > 0.0,
                escape_term,
                lambda _: jnp.array(0.0, dtype=x.dtype),
                operand=None,
            )
            total = main_loss + jac_weight_now * l_jac + escape_weight_now * l_escape
            return total, (l_vf, l_res, l_vf_norm, l_res_norm, l_res_obj, l_jac, l_escape)

        (
            loss_value,
            (l_vf, l_res, l_vf_norm, l_res_norm, l_res_obj, l_jac, l_escape),
        ), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)
        grads = eqx.filter(grads, eqx.is_array)
        model_ = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, model_)
        model = eqx.apply_updates(model, updates)
        return (
            model,
            opt_state,
            loss_value,
            l_vf,
            l_res,
            l_vf_norm,
            l_res_norm,
            l_res_obj,
            l_jac,
            l_escape,
        )

    def finite_output_ratio(model, x):
        """Fraction of finite model outputs over a batch."""
        y_pred = model(x)
        return jnp.mean(jnp.isfinite(y_pred))

    def blend_beta(epoch):
        """Linear blend coefficient from VF loss to residual loss."""
        if epoch < stage0_epochs:
            return beta_start
        if epoch >= blend_end_epochs:
            return 1.0
        progress = (epoch - stage0_epochs) / max(1, (blend_end_epochs - stage0_epochs))
        return beta_start + (1.0 - beta_start) * progress

    def jac_weight_at_epoch(epoch):
        """Piecewise-linear schedule for Jacobian regularization weight."""
        if epoch < jacobian_start_epoch:
            return 0.0
        ramp = (epoch - jacobian_start_epoch + 1) / jacobian_ramp_epochs
        return jacobian_weight * min(1.0, ramp)

    def escape_gamma_at_epoch(epoch):
        """Residual amplification factor during the escape phase."""
        if escape_epochs <= 0:
            return 1.0
        if epoch >= escape_epochs:
            return escape_gamma_end
        progress = epoch / max(1, escape_epochs - 1)
        return escape_gamma_start + progress * (escape_gamma_end - escape_gamma_start)

    def escape_weight_at_epoch(epoch):
        """Decaying strength for strong-field escape regularization."""
        if (escape_epochs <= 0) or (epoch >= escape_epochs):
            return 0.0
        return escape_weight * (1.0 - (epoch / max(1, escape_epochs)))

    def sample_epoch_batch_indices(key_epoch, epoch):
        """Sample fixed-shape batch indices, with optional high-u oversampling."""
        escape_sampling_active = (
            (epoch < escape_epochs)
            and (escape_high_u_frac > 0.0)
            and (high_pool_idx.shape[0] > 0)
        )
        if not escape_sampling_active:
            perm = jax.random.permutation(key_epoch, num_train_samples)
            return perm[:used_train_samples].reshape((num_train_batches, effective_batch_size))

        hi_count = int(round(escape_high_u_frac * effective_batch_size))
        hi_count = max(1, min(hi_count, effective_batch_size))
        lo_count = effective_batch_size - hi_count
        lo_pool = low_pool_idx if low_pool_idx.shape[0] > 0 else all_pool_idx

        key_now = key_epoch
        batches = []
        for _ in range(num_train_batches):
            key_now, key_hi, key_lo, key_shuffle = jax.random.split(key_now, 4)
            hi_idx = jax.random.choice(key_hi, high_pool_idx, shape=(hi_count,), replace=True)
            if lo_count > 0:
                lo_idx = jax.random.choice(key_lo, lo_pool, shape=(lo_count,), replace=True)
                batch_idx = jnp.concatenate([hi_idx, lo_idx], axis=0)
            else:
                batch_idx = hi_idx
            batch_idx = jax.random.permutation(key_shuffle, batch_idx)
            batches.append(batch_idx)
        return jnp.stack(batches, axis=0)

    for epoch in range(training_params["adam_epochs"]):
        beta = blend_beta(epoch)
        jac_w = jac_weight_at_epoch(epoch)
        esc_gamma = escape_gamma_at_epoch(epoch)
        esc_w = escape_weight_at_epoch(epoch)
        beta_arr = jnp.array(beta, dtype=x_train.dtype)
        jac_w_arr = jnp.array(jac_w, dtype=x_train.dtype)
        esc_gamma_arr = jnp.array(esc_gamma, dtype=x_train.dtype)
        esc_w_arr = jnp.array(esc_w, dtype=x_train.dtype)
        esc_u_thresh_arr = jnp.array(escape_u_threshold, dtype=x_train.dtype)
        esc_margin_arr = jnp.array(escape_margin, dtype=x_train.dtype)

        key, key_train = jax.random.split(key, 2)
        batch_indices = sample_epoch_batch_indices(key_train, epoch)

        epoch_loss = 0.0
        epoch_l_vf = 0.0
        epoch_l_res = 0.0
        epoch_l_vf_norm = 0.0
        epoch_l_res_norm = 0.0
        epoch_l_res_obj = 0.0
        epoch_l_jac = 0.0
        epoch_l_escape = 0.0
        for batch_idx in batch_indices:
            x_train_batch = jnp.take(x_train, batch_idx, axis=0)
            y_train_batch = jnp.take(y_train, batch_idx, axis=0)
            y_base_batch = jnp.take(y_base_train, batch_idx, axis=0)
            batch_bin_idx = jnp.take(train_bin_idx, batch_idx, axis=0)
            residual_scales_batch = jnp.take(residual_scale_table, batch_bin_idx, axis=0)
            x_jac_batch = x_train_batch[:jacobian_batch_size]
            (
                model,
                opt_state,
                batch_loss,
                batch_l_vf,
                batch_l_res,
                batch_l_vf_norm,
                batch_l_res_norm,
                batch_l_res_obj,
                batch_l_jac,
                batch_l_escape,
            ) = step(
                model,
                opt_state,
                x_train_batch,
                y_train_batch,
                y_base_batch,
                residual_scales_batch,
                x_jac_batch,
                vf_scales,
                vf_ref,
                res_ref,
                beta_arr,
                jac_w_arr,
                esc_gamma_arr,
                esc_w_arr,
                esc_u_thresh_arr,
                esc_margin_arr,
            )
            epoch_loss += batch_loss
            epoch_l_vf += batch_l_vf
            epoch_l_res += batch_l_res
            epoch_l_vf_norm += batch_l_vf_norm
            epoch_l_res_norm += batch_l_res_norm
            epoch_l_res_obj += batch_l_res_obj
            epoch_l_jac += batch_l_jac
            epoch_l_escape += batch_l_escape

        loss_value = epoch_loss / num_train_batches
        train_l_vf = epoch_l_vf / num_train_batches
        train_l_res = epoch_l_res / num_train_batches
        train_l_vf_norm = epoch_l_vf_norm / num_train_batches
        train_l_res_norm = epoch_l_res_norm / num_train_batches
        train_l_res_obj = epoch_l_res_obj / num_train_batches
        train_l_jac = epoch_l_jac / num_train_batches
        train_l_escape = epoch_l_escape / num_train_batches

        y_val_pred = model(x_val)
        delta_val_pred = y_val_pred - y_base_val
        delta_val_true = y_val - y_base_val
        val_l_vf = _standardized_vf_loss_from_pred(y_val_pred, y_val, vf_scales, eps)
        val_l_res = _standardized_residual_loss_from_delta(
            delta_val_pred, delta_val_true, val_residual_scales, eps
        )
        val_l_vf_norm = val_l_vf / vf_ref
        val_l_res_norm = val_l_res / res_ref
        val_l_res_obj = esc_gamma_arr * val_l_res_norm
        val_loss_value = (1.0 - beta_arr) * val_l_vf_norm + beta_arr * val_l_res_obj

        if epoch % 10 == 0:
            val_finite_ratio = finite_output_ratio(model, x_val)
            print(
                f"Epoch {epoch}, Loss: {loss_value}, Val Loss: {val_loss_value}, "
                f"Train VF: {train_l_vf}, Train RES: {train_l_res}, Train JAC: {train_l_jac}, "
                f"Train ESC: {train_l_escape}, "
                f"Train VF*: {train_l_vf_norm}, Train RES*: {train_l_res_norm}, "
                f"Train RESobj*: {train_l_res_obj}, "
                f"Val VF: {val_l_vf}, Val RES: {val_l_res}, "
                f"Val VF*: {val_l_vf_norm}, Val RES*: {val_l_res_norm}, "
                f"Val RESobj*: {val_l_res_obj}, Beta: {beta:.3f}, "
                f"JacW: {jac_w:.2e}, EscG: {esc_gamma:.2f}, EscW: {esc_w:.2e}, "
                f"Val finite ratio: {val_finite_ratio}"
            )
        if val_loss_value < 1e-6:
            return model

    return model


def train_blackbox_dhnn_model_prelim(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    val_data: Tuple[jnp.ndarray, jnp.ndarray],
    model_params: Dict[str, int],
    training_params: Dict[str, Union[int, float]],
) -> BlackBoxDHNN:
    """Train a black-box DHNN baseline on pure vector-field supervision."""

    model_class = model_params.get("model_class", BlackBoxDHNN)
    model_kwargs = {k: v for k, v in model_params.items() if k != "model_class"}
    key = model_kwargs["key"]
    model = model_class(**model_kwargs)

    x_train, y_train, _ = _split_supervision_data(train_data, "train")
    x_val, y_val, _ = _split_supervision_data(val_data, "val")
    batch_size = int(training_params["batch_size"])
    strict_target_validation = bool(training_params.get("strict_target_validation", True))

    _validate_rhs_targets(y_train, "train", strict=strict_target_validation)
    _validate_rhs_targets(y_val, "val", strict=strict_target_validation)

    num_train_samples = x_train.shape[0]
    effective_batch_size = min(batch_size, num_train_samples)
    num_train_batches = max(1, num_train_samples // effective_batch_size)
    used_train_samples = num_train_batches * effective_batch_size
    dropped_train_samples = num_train_samples - used_train_samples

    lr_peak = float(training_params.get("learning_rate", 3e-4))
    lr_init = float(training_params.get("lr_init", 0.1 * lr_peak))
    lr_end = float(training_params.get("lr_end", 0.01 * lr_peak))
    total_steps = int(training_params["adam_epochs"]) * num_train_batches
    warmup_steps_requested = int(training_params.get("warmup_steps", int(0.1 * total_steps)))
    warmup_steps = min(max(0, warmup_steps_requested), max(0, total_steps - 1))
    decay_steps = max(total_steps, warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_init,
        peak_value=lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_end,
    )
    print(
        "BlackBox LR schedule:",
        {
            "init": lr_init,
            "peak": lr_peak,
            "end": lr_end,
            "warmup_steps": warmup_steps,
            "warmup_steps_requested": warmup_steps_requested,
            "total_steps": total_steps,
            "decay_steps": decay_steps,
        },
    )

    eps = float(training_params.get("loss_eps", 1e-8))
    vf_scales = _component_scales(y_train, eps)
    vf_ref = jnp.maximum(
        _standardized_vf_loss_from_pred(model(x_train), y_train, vf_scales, eps), eps
    )
    print("BlackBox VF scales:", vf_scales)
    print("BlackBox VF anchor:", vf_ref)
    if dropped_train_samples > 0:
        print(
            f"Dropping {dropped_train_samples} train samples each epoch "
            f"to keep a fixed JIT batch shape."
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y, vf_scales, vf_ref):
        def loss_fn(m):
            y_pred = m(x)
            l_vf = _standardized_vf_loss_from_pred(y_pred, y, vf_scales, eps)
            l_vf_norm = l_vf / vf_ref
            return l_vf_norm, (l_vf, l_vf_norm)

        (loss_value, (l_vf, l_vf_norm)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)
        grads = eqx.filter(grads, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, l_vf, l_vf_norm

    def finite_output_ratio(model, x):
        y_pred = model(x)
        return jnp.mean(jnp.isfinite(y_pred))

    for epoch in range(int(training_params["adam_epochs"])):
        key, key_train = jax.random.split(key, 2)
        perm = jax.random.permutation(key_train, num_train_samples)
        batch_indices = perm[:used_train_samples].reshape((num_train_batches, effective_batch_size))

        epoch_loss = 0.0
        epoch_l_vf = 0.0
        epoch_l_vf_norm = 0.0
        for batch_idx in batch_indices:
            x_batch = jnp.take(x_train, batch_idx, axis=0)
            y_batch = jnp.take(y_train, batch_idx, axis=0)
            model, opt_state, batch_loss, batch_l_vf, batch_l_vf_norm = step(
                model, opt_state, x_batch, y_batch, vf_scales, vf_ref
            )
            epoch_loss += batch_loss
            epoch_l_vf += batch_l_vf
            epoch_l_vf_norm += batch_l_vf_norm

        train_loss = epoch_loss / num_train_batches
        train_l_vf = epoch_l_vf / num_train_batches
        train_l_vf_norm = epoch_l_vf_norm / num_train_batches

        y_val_pred = model(x_val)
        val_l_vf = _standardized_vf_loss_from_pred(y_val_pred, y_val, vf_scales, eps)
        val_loss = val_l_vf / vf_ref

        if epoch % 10 == 0:
            val_finite_ratio = finite_output_ratio(model, x_val)
            print(
                f"[BlackBox] Epoch {epoch}, Loss: {train_loss}, Val Loss: {val_loss}, "
                f"Train VF: {train_l_vf}, Train VF*: {train_l_vf_norm}, "
                f"Val VF: {val_l_vf}, Val VF*: {val_loss}, "
                f"Val finite ratio: {val_finite_ratio}"
            )

    return model


def train_hybrid_eob_dhnn_model_prelim(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    val_data: Tuple[jnp.ndarray, jnp.ndarray],
    model_params: Dict[str, int],
    training_params: Dict[str, Union[int, float]],
) -> Hybrid_EOB_DHNN:
    """Train hybrid EOB-form DHNN with potential-aware RHS channel losses."""

    model_class = model_params.get("model_class", Hybrid_EOB_DHNN)
    model_kwargs = {k: v for k, v in model_params.items() if k != "model_class"}
    key = model_kwargs["key"]
    model = model_class(**model_kwargs)
    load_weights_path = str(training_params.get("load_weights_path", "")).strip()
    load_weights_strict = bool(training_params.get("load_weights_strict", True))
    save_weights_path = str(training_params.get("save_weights_path", "")).strip()
    if load_weights_path:
        model = load_model_weights(model, load_weights_path, strict=load_weights_strict)

    x_train, y_train, _ = _split_supervision_data(train_data, "train")
    x_val, y_val, e_rel_val_ref = _split_supervision_data(val_data, "val")
    batch_size = int(training_params["batch_size"])
    strict_target_validation = bool(training_params.get("strict_target_validation", True))

    _validate_rhs_targets(y_train, "train", strict=strict_target_validation)
    _validate_rhs_targets(y_val, "val", strict=strict_target_validation)

    num_train_samples = x_train.shape[0]
    effective_batch_size = min(batch_size, num_train_samples)
    num_train_batches = max(1, num_train_samples // effective_batch_size)
    used_train_samples = num_train_batches * effective_batch_size
    dropped_train_samples = num_train_samples - used_train_samples

    lr_peak = float(training_params.get("learning_rate", 3e-4))
    lr_init = float(training_params.get("lr_init", 0.1 * lr_peak))
    lr_end = float(training_params.get("lr_end", 0.01 * lr_peak))
    total_steps = int(training_params["adam_epochs"]) * num_train_batches
    warmup_steps_requested = int(training_params.get("warmup_steps", int(0.1 * total_steps)))
    warmup_steps = min(max(0, warmup_steps_requested), max(0, total_steps - 1))
    decay_steps = max(total_steps, warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_init,
        peak_value=lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_end,
    )
    print(
        "HybridEOB LR schedule:",
        {
            "init": lr_init,
            "peak": lr_peak,
            "end": lr_end,
            "warmup_steps": warmup_steps,
            "warmup_steps_requested": warmup_steps_requested,
            "total_steps": total_steps,
            "decay_steps": decay_steps,
        },
    )

    eps = float(training_params.get("loss_eps", 1e-8))
    vf_scales = _component_scales(y_train, eps)
    vf_ref = jnp.maximum(
        _standardized_vf_loss_from_pred(model(x_train), y_train, vf_scales, eps), eps
    )

    p_phi_train_safe = jnp.where(
        jnp.abs(x_train[:, 4]) < 1e-12,
        jnp.where(x_train[:, 4] >= 0.0, 1e-12, -1e-12),
        x_train[:, 4],
    )
    cons_target_train = y_train[:, 2] - y_train[:, 3] * (x_train[:, 3] / p_phi_train_safe)
    cons_scale = jnp.maximum(jnp.std(cons_target_train), eps)

    p_r_train_safe = jnp.where(
        jnp.abs(x_train[:, 3]) < 1e-12,
        jnp.where(x_train[:, 3] >= 0.0, 1e-12, -1e-12),
        x_train[:, 3],
    )
    dr_over_pr_train = y_train[:, 0] / p_r_train_safe
    q_scale = jnp.maximum(jnp.std(dr_over_pr_train), eps)

    w_vf = float(training_params.get("hybrid_w_vf", 1.0))
    w_flux = float(training_params.get("hybrid_w_flux", 1.0))
    w_omega = float(training_params.get("hybrid_w_omega", 1.0))
    w_cons = float(training_params.get("hybrid_w_cons", 1.0))
    w_q = float(training_params.get("hybrid_w_q", 0.5))
    use_adaptive_pr_masks = bool(training_params.get("hybrid_use_adaptive_pr_masks", True))
    qc_frac_target = float(training_params.get("hybrid_qc_frac_target", 0.15))
    q_frac_target = float(training_params.get("hybrid_q_frac_target", 0.80))
    qc_frac_target = min(max(qc_frac_target, 1e-4), 0.5)
    q_frac_target = min(max(q_frac_target, qc_frac_target + 1e-4), 0.999)
    if use_adaptive_pr_masks:
        abs_pr_train_np = np.asarray(jnp.abs(x_train[:, 3]))
        pr_qc_threshold = float(np.quantile(abs_pr_train_np, qc_frac_target))
        pr_q_threshold = float(np.quantile(abs_pr_train_np, q_frac_target))
    else:
        pr_qc_threshold = float(training_params.get("hybrid_pr_qc_threshold", 2e-3))
        pr_q_threshold = float(training_params.get("hybrid_pr_q_threshold", 8e-3))
    pr_qc_threshold = max(pr_qc_threshold, float(eps))
    pr_q_threshold = max(pr_q_threshold, pr_qc_threshold + float(eps))
    q_start_epoch = int(training_params.get("hybrid_q_start_epoch", 200))
    q_ramp_epochs = max(1, int(training_params.get("hybrid_q_ramp_epochs", 200)))
    vf_only_min_epochs = int(training_params.get("hybrid_vf_only_min_epochs", 50))
    vf_only_max_epochs = int(training_params.get("hybrid_vf_only_max_epochs", 400))
    vf_only_target_vf = float(training_params.get("hybrid_vf_only_target_vf", 0.25))
    geom_ramp_epochs = max(1, int(training_params.get("hybrid_geom_ramp_epochs", 100)))
    stop_on_rel_err = bool(training_params.get("hybrid_stop_on_rel_err", False))
    rel_err_target = float(training_params.get("hybrid_rel_err_target", 1e-3))
    rel_err_min_epochs = int(training_params.get("hybrid_rel_err_min_epochs", 100))
    use_val_erel_threshold = bool(training_params.get("hybrid_use_val_erel_threshold", True))
    stop_on_pert_ratio = bool(training_params.get("hybrid_stop_on_pert_ratio", False))
    pert_alpha = float(training_params.get("hybrid_pert_alpha", 10.0))
    pert_metric = str(training_params.get("hybrid_pert_metric", "p95")).lower()
    if pert_metric not in {"mean", "p95"}:
        pert_metric = "p95"
    pert_componentwise = bool(training_params.get("hybrid_pert_componentwise", False))
    pert_min_epochs = int(training_params.get("hybrid_pert_min_epochs", rel_err_min_epochs))
    num_r_bins = max(1, int(training_params.get("hybrid_num_r_bins", 8)))
    r_bins_mode = str(training_params.get("hybrid_r_bins_mode", "linear")).lower()
    log_r_binned_val = bool(training_params.get("hybrid_log_r_binned_val", True))
    r_binned_top_k = max(1, int(training_params.get("hybrid_r_binned_top_k", 2)))
    r_binned_sort_key = str(training_params.get("hybrid_r_binned_sort_key", "cons")).lower()
    valid_bin_sort_keys = {"vf", "flux", "omega", "cons", "q"}
    if r_binned_sort_key not in valid_bin_sort_keys:
        r_binned_sort_key = "cons"

    print("HybridEOB VF scales:", vf_scales)
    print("HybridEOB VF anchor:", vf_ref)
    train_abs_pr = jnp.abs(x_train[:, 3])
    train_qc_frac = jnp.mean(train_abs_pr <= pr_qc_threshold)
    train_q_frac = jnp.mean(train_abs_pr > pr_q_threshold)
    print(
        "HybridEOB channel scales:",
        {
            "cons_scale": cons_scale,
            "q_scale": q_scale,
            "pr_qc_threshold": pr_qc_threshold,
            "pr_q_threshold": pr_q_threshold,
            "train_qc_frac": train_qc_frac,
            "train_q_frac": train_q_frac,
            "adaptive_pr_masks": use_adaptive_pr_masks,
        },
    )
    print(
        "HybridEOB loss weights:",
        {
            "w_vf": w_vf,
            "w_flux": w_flux,
            "w_omega": w_omega,
            "w_cons": w_cons,
            "w_q": w_q,
            "q_start_epoch": q_start_epoch,
            "q_ramp_epochs": q_ramp_epochs,
            "vf_only_min_epochs": vf_only_min_epochs,
            "vf_only_max_epochs": vf_only_max_epochs,
            "vf_only_target_vf": vf_only_target_vf,
            "geom_ramp_epochs": geom_ramp_epochs,
            "stop_on_rel_err": stop_on_rel_err,
            "rel_err_target": rel_err_target,
            "rel_err_min_epochs": rel_err_min_epochs,
            "use_val_erel_threshold": use_val_erel_threshold,
            "stop_on_pert_ratio": stop_on_pert_ratio,
            "pert_alpha": pert_alpha,
            "pert_metric": pert_metric,
            "pert_componentwise": pert_componentwise,
            "pert_min_epochs": pert_min_epochs,
        },
    )
    pert_ref_summary = None
    if (e_rel_val_ref is not None) and use_val_erel_threshold:
        e_rel_val_ref = jnp.asarray(e_rel_val_ref)
        if e_rel_val_ref.shape != y_val.shape:
            raise ValueError(
                f"Validation perturbation reference shape mismatch: "
                f"e_rel_val_ref={tuple(e_rel_val_ref.shape)} vs y_val={tuple(y_val.shape)}."
            )
        pert_ref_summary = _relative_error_summary(e_rel_val_ref)
        print(
            "HybridEOB validation perturbation reference:",
            {
                "mean": pert_ref_summary["mean"],
                "p95": pert_ref_summary["p95"],
                "finite_ratio": pert_ref_summary["finite_ratio"],
                "comp_mean": pert_ref_summary["comp_mean"],
                "comp_p95": pert_ref_summary["comp_p95"],
            },
        )
    r_abs_train_np = np.asarray(jnp.abs(x_train[:, 1]))
    r_abs_min = float(np.min(r_abs_train_np))
    r_abs_max = float(np.max(r_abs_train_np))
    if r_bins_mode == "quantile":
        r_bin_edges = np.quantile(r_abs_train_np, np.linspace(0.0, 1.0, num_r_bins + 1))
        # Guard against repeated quantile edges from finite precision.
        for i in range(1, r_bin_edges.shape[0]):
            if r_bin_edges[i] <= r_bin_edges[i - 1]:
                r_bin_edges[i] = r_bin_edges[i - 1] + 1e-12
    else:
        r_bin_edges = np.linspace(r_abs_min, r_abs_max, num_r_bins + 1)
    print(
        "HybridEOB r-bin config:",
        {
            "mode": r_bins_mode,
            "num_r_bins": num_r_bins,
            "r_min": r_abs_min,
            "r_max": r_abs_max,
            "log_r_binned_val": log_r_binned_val,
            "top_k": r_binned_top_k,
            "sort_key": r_binned_sort_key,
        },
    )
    if dropped_train_samples > 0:
        print(
            f"Dropping {dropped_train_samples} train samples each epoch "
            f"to keep a fixed JIT batch shape."
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def hybrid_loss_terms(m, x, y, q_gain, geom_gain):
        y_pred = m(x)
        l_vf = _standardized_vf_loss_from_pred(y_pred, y, vf_scales, eps)
        l_vf_norm = l_vf / vf_ref
        rel_abs_mean, rel_abs_comp = _componentwise_relative_error_metrics_from_pred(y_pred, y, eps=eps)

        flux_true = y[:, 3]
        flux_pred = y_pred[:, 3]
        l_flux = jnp.mean((_safe_log_abs(flux_pred, eps) - _safe_log_abs(flux_true, eps)) ** 2)

        omega_true = jnp.maximum(y[:, 1], eps)
        omega_pred = jnp.maximum(y_pred[:, 1], eps)
        p_r = x[:, 3]
        p_phi = x[:, 4]
        abs_pr = jnp.abs(p_r)
        mask_rad = abs_pr > pr_qc_threshold
        mask_q = abs_pr > pr_q_threshold

        # Smooth quasi-circular weighting prevents the omega channel from
        # collapsing to zero when a hard mask is sparsely populated.
        qc_scale = jnp.maximum(pr_qc_threshold, eps)
        w_qc = jnp.exp(-jnp.square(abs_pr / qc_scale))
        omega_err_sq = (jnp.log(omega_pred) - jnp.log(omega_true)) ** 2
        l_omega = jnp.sum(w_qc * omega_err_sq) / jnp.maximum(jnp.sum(w_qc), 1.0)

        p_phi_safe = jnp.where(
            jnp.abs(p_phi) < 1e-12,
            jnp.where(p_phi >= 0.0, 1e-12, -1e-12),
            p_phi,
        )
        cons_true = y[:, 2] - flux_true * (p_r / p_phi_safe)
        cons_pred = y_pred[:, 2] - flux_pred * (p_r / p_phi_safe)
        l_cons = _masked_mean(((cons_pred - cons_true) / (cons_scale + eps)) ** 2, mask_rad)

        p_r_safe = jnp.where(
            jnp.abs(p_r) < 1e-12,
            jnp.where(p_r >= 0.0, 1e-12, -1e-12),
            p_r,
        )
        dr_ratio_true = y[:, 0] / p_r_safe
        dr_ratio_pred = y_pred[:, 0] / p_r_safe
        l_q_raw = _masked_mean(((dr_ratio_pred - dr_ratio_true) / (q_scale + eps)) ** 2, mask_q)
        l_q = q_gain * l_q_raw

        l_total = (
            w_vf * l_vf_norm
            + geom_gain * w_flux * l_flux
            + geom_gain * w_omega * l_omega
            + geom_gain * w_cons * l_cons
            + w_q * l_q
        )
        l_total = l_vf
        return l_total, (l_vf, l_vf_norm, l_flux, l_omega, l_cons, l_q_raw, rel_abs_mean, rel_abs_comp)

    @eqx.filter_jit
    def step(model, opt_state, x, y, q_gain, geom_gain):
        def loss_fn(m):
            return hybrid_loss_terms(m, x, y, q_gain, geom_gain)

        (loss_value, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        grads = eqx.filter(grads, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        l_vf, l_vf_norm, l_flux, l_omega, l_cons, l_q_raw, rel_abs_mean, rel_abs_comp = aux
        return (
            model,
            opt_state,
            loss_value,
            l_vf,
            l_vf_norm,
            l_flux,
            l_omega,
            l_cons,
            l_q_raw,
            rel_abs_mean,
            rel_abs_comp,
        )

    def finite_output_ratio(model, x):
        y_pred = model(x)
        return jnp.mean(jnp.isfinite(y_pred))

    def r_binned_val_metrics(model, x_val, y_val, q_gain, geom_gain):
        r_abs_val_np = np.asarray(jnp.abs(x_val[:, 1]))
        rows = []
        for b in range(num_r_bins):
            lo = float(r_bin_edges[b])
            hi = float(r_bin_edges[b + 1])
            if b < (num_r_bins - 1):
                mask = (r_abs_val_np >= lo) & (r_abs_val_np < hi)
            else:
                mask = (r_abs_val_np >= lo) & (r_abs_val_np <= hi)
            idx_np = np.nonzero(mask)[0]
            count = int(idx_np.shape[0])
            if count == 0:
                continue

            idx = jnp.asarray(idx_np, dtype=jnp.int32)
            x_bin = jnp.take(x_val, idx, axis=0)
            y_bin = jnp.take(y_val, idx, axis=0)
            _, aux_bin = hybrid_loss_terms(model, x_bin, y_bin, q_gain, geom_gain)
            l_vf_b, l_vf_norm_b, l_flux_b, l_omega_b, l_cons_b, l_q_b, *_ = aux_bin
            rows.append(
                {
                    "bin": b,
                    "r_lo": lo,
                    "r_hi": hi,
                    "n": count,
                    "vf": float(l_vf_b),
                    "vf_norm": float(l_vf_norm_b),
                    "flux": float(l_flux_b),
                    "omega": float(l_omega_b),
                    "cons": float(l_cons_b),
                    "q": float(l_q_b),
                }
            )
        return rows

    def compact_r_binned_summary(rows):
        if len(rows) == 0:
            return {"weighted": {}, "worst": []}
        metric_keys = ("vf", "flux", "omega", "cons", "q")
        n_total = float(sum(row["n"] for row in rows))
        n_total = max(n_total, 1.0)
        weighted = {
            key: float(sum(row[key] * row["n"] for row in rows) / n_total)
            for key in metric_keys
        }
        worst_rows = sorted(rows, key=lambda row: row[r_binned_sort_key], reverse=True)[:r_binned_top_k]
        worst_compact = [
            {
                "bin": row["bin"],
                "r_lo": row["r_lo"],
                "r_hi": row["r_hi"],
                "n": row["n"],
                r_binned_sort_key: row[r_binned_sort_key],
            }
            for row in worst_rows
        ]
        return {"weighted": weighted, "worst": worst_compact}

    geom_unlocked = False
    geom_unlock_epoch = -1
    last_val_l_vf = jnp.array(jnp.inf, dtype=x_train.dtype)
    last_val_rel_summary = None
    last_val_pert_ratio = np.inf
    last_val_pert_ratio_comp = None

    for epoch in range(int(training_params["adam_epochs"])):
        key, key_train = jax.random.split(key, 2)
        perm = jax.random.permutation(key_train, num_train_samples)
        batch_indices = perm[:used_train_samples].reshape((num_train_batches, effective_batch_size))

        if (not geom_unlocked) and (epoch >= vf_only_min_epochs):
            if (float(last_val_l_vf) <= vf_only_target_vf) or (epoch >= vf_only_max_epochs):
                geom_unlocked = True
                geom_unlock_epoch = epoch
                print(
                    f"[HybridEOB] Geometric-channel unlock at epoch {epoch} "
                    f"(last_val_vf={float(last_val_l_vf):.6g})"
                )

        if geom_unlocked:
            geom_gain = min(1.0, (epoch - geom_unlock_epoch + 1) / float(geom_ramp_epochs))
        else:
            geom_gain = 0.0

        if epoch < q_start_epoch:
            q_gain_raw = 0.0
        else:
            q_gain_raw = min(1.0, (epoch - q_start_epoch + 1) / float(q_ramp_epochs))
        q_gain = geom_gain * q_gain_raw

        epoch_loss = 0.0
        epoch_l_vf = 0.0
        epoch_l_vf_norm = 0.0
        epoch_l_flux = 0.0
        epoch_l_omega = 0.0
        epoch_l_cons = 0.0
        epoch_l_q = 0.0
        epoch_rel_abs_mean = 0.0
        epoch_rel_abs_comp = jnp.zeros((y_train.shape[1],), dtype=y_train.dtype)
        for batch_idx in batch_indices:
            x_batch = jnp.take(x_train, batch_idx, axis=0)
            y_batch = jnp.take(y_train, batch_idx, axis=0)
            (
                model,
                opt_state,
                batch_loss,
                batch_l_vf,
                batch_l_vf_norm,
                batch_l_flux,
                batch_l_omega,
                batch_l_cons,
                batch_l_q,
                batch_rel_abs_mean,
                batch_rel_abs_comp,
            ) = step(model, opt_state, x_batch, y_batch, q_gain, geom_gain)
            epoch_loss += batch_loss
            epoch_l_vf += batch_l_vf
            epoch_l_vf_norm += batch_l_vf_norm
            epoch_l_flux += batch_l_flux
            epoch_l_omega += batch_l_omega
            epoch_l_cons += batch_l_cons
            epoch_l_q += batch_l_q
            epoch_rel_abs_mean += batch_rel_abs_mean
            epoch_rel_abs_comp += batch_rel_abs_comp

        train_loss = epoch_loss / num_train_batches
        train_l_vf = epoch_l_vf / num_train_batches
        train_l_vf_norm = epoch_l_vf_norm / num_train_batches
        train_l_flux = epoch_l_flux / num_train_batches
        train_l_omega = epoch_l_omega / num_train_batches
        train_l_cons = epoch_l_cons / num_train_batches
        train_l_q = epoch_l_q / num_train_batches
        train_rel_abs_mean = epoch_rel_abs_mean / num_train_batches
        train_rel_abs_comp = epoch_rel_abs_comp / num_train_batches

        val_loss, val_aux = hybrid_loss_terms(model, x_val, y_val, q_gain, geom_gain)
        (
            val_l_vf,
            val_l_vf_norm,
            val_l_flux,
            val_l_omega,
            val_l_cons,
            val_l_q,
            val_rel_abs_mean,
            val_rel_abs_comp,
        ) = val_aux
        last_val_l_vf = val_l_vf

        # Validation-only threshold metric: compare NN relative error to SEOB perturbation floor.
        y_val_pred_for_rel = model(x_val)
        val_rel_matrix = _relative_error_matrix_from_pred(y_val_pred_for_rel, y_val, eps=eps)
        last_val_rel_summary = _relative_error_summary(val_rel_matrix)
        if pert_ref_summary is not None:
            if pert_metric == "p95":
                denom = max(float(pert_ref_summary["p95"]), eps)
                numer = float(last_val_rel_summary["p95"])
                denom_comp = np.maximum(np.asarray(pert_ref_summary["comp_p95"]), eps)
                numer_comp = np.asarray(last_val_rel_summary["comp_p95"])
            else:
                denom = max(float(pert_ref_summary["mean"]), eps)
                numer = float(last_val_rel_summary["mean"])
                denom_comp = np.maximum(np.asarray(pert_ref_summary["comp_mean"]), eps)
                numer_comp = np.asarray(last_val_rel_summary["comp_mean"])
            last_val_pert_ratio = numer / denom
            last_val_pert_ratio_comp = numer_comp / denom_comp

        if epoch % 10 == 0:
            val_finite_ratio = finite_output_ratio(model, x_val)
            print(
                f"[HybridEOB] Epoch {epoch}, Loss: {train_loss}, Val Loss: {val_loss}, "
                f"Val VF: {val_l_vf}, Val Flux: {val_l_flux}, Val Omega: {val_l_omega}, "
                f"Val Cons: {val_l_cons}, Val Q: {val_l_q}, "
                f"ValRel(mean,p95)=({last_val_rel_summary['mean']:.3e}, {last_val_rel_summary['p95']:.3e}), "
                f"Val finite ratio: {val_finite_ratio}"
#                f"Train VF: {train_l_vf}, Train VF*: {train_l_vf_norm}, "
#                f"Train Flux: {train_l_flux}, Train Omega: {train_l_omega}, "
#                f"Train Cons: {train_l_cons}, Train Q: {train_l_q}, "
#                f"Train RelAbs: {train_rel_abs_mean}, "
#                f"Val VF: {val_l_vf}, Val VF*: {val_l_vf_norm}, "
#                f"Val Flux: {val_l_flux}, Val Omega: {val_l_omega}, "
#                f"Val Cons: {val_l_cons}, Val Q: {val_l_q}, "
#                f"Val RelAbs: {val_rel_abs_mean}, "
#                f"GeomGain: {geom_gain:.3f}, QGain: {q_gain:.3f}, "
#                f"Val finite ratio: {val_finite_ratio}"
            )
            if pert_ref_summary is not None:
                print(
                    f"[HybridEOB] Val/SEOB-pert ratio ({pert_metric}): "
                    f"{last_val_pert_ratio:.3e} (target <= {pert_alpha:.3e})"
                )
                print(
                    "[HybridEOB] Val/SEOB-pert component ratios:",
                    np.asarray(last_val_pert_ratio_comp),
                )
            print(
                "HybridEOB RelAbsComp(train,val):",
                np.asarray(train_rel_abs_comp),
                np.asarray(val_rel_abs_comp),
            )
            if log_r_binned_val:
                binned_rows = r_binned_val_metrics(model, x_val, y_val, q_gain, geom_gain)
#                print("HybridEOB Val r-bin summary:", compact_r_binned_summary(binned_rows))

        if stop_on_rel_err and (epoch >= rel_err_min_epochs) and (float(val_rel_abs_mean) <= rel_err_target):
            print(
                f"[HybridEOB] Early stop by relative-error threshold at epoch {epoch}: "
                f"val_rel_abs={float(val_rel_abs_mean):.6g} <= target={rel_err_target:.6g}"
            )
            if save_weights_path:
                save_model_weights(model, save_weights_path)
            return model
        if (
            stop_on_pert_ratio
            and (pert_ref_summary is not None)
            and (epoch >= pert_min_epochs)
        ):
            if pert_componentwise:
                cond = bool(np.all(np.asarray(last_val_pert_ratio_comp) <= pert_alpha))
            else:
                cond = bool(last_val_pert_ratio <= pert_alpha)
            if cond:
                print(
                    f"[HybridEOB] Early stop by SEOB-perturbation threshold at epoch {epoch}: "
                    f"val/pert_{pert_metric}={last_val_pert_ratio:.6g} <= alpha={pert_alpha:.6g}"
                )
                if save_weights_path:
                    save_model_weights(model, save_weights_path)
                return model

    if save_weights_path:
        save_model_weights(model, save_weights_path)
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # training parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    training_params = {
        "experiment": "hybrid_eob",
        "learning_rate": 3e-4,
        "lr_init": 3e-5,
        "lr_end": 3e-6,
        "adam_epochs": 5000,
        "lbfgs_epochs": 5000,
        "batch_size": 1000,
        "stage0_epochs": 500,
        "blend_end_epochs": 2000,
        "beta_start": 0.05,
        "warmup_steps": 2000,
        "jacobian_start_epoch": 600,
        "jacobian_ramp_epochs": 300,
        "jacobian_weight": 1e-5,
        "jacobian_batch_size": 16,
        "num_u_bins": 8,
        "min_bin_count": 32,
        "residual_scale_floor_frac": 0.1,
        "escape_epochs": 80,
        "escape_gamma_start": 1.2,
        "escape_gamma_end": 1.0,
        "escape_weight": 0.0,
        "escape_margin": 0.1,
        "escape_u_threshold": 0.12,
        "escape_high_u_frac": 0.1,
        "escape_top_bins": 1,
        "loss_eps": 1e-8,
        "strict_target_validation": True,
        "load_weights_path": "",
        "load_weights_strict": True,
        "save_weights_path": "hybrid_eob_weights.eqx",
        "hybrid_w_vf": 1.0,
        "hybrid_w_flux": 1.0,
        "hybrid_w_omega": 1.0,
        "hybrid_w_cons": 1.0,
        "hybrid_w_q": 0.5,
        "hybrid_use_adaptive_pr_masks": True,
        "hybrid_qc_frac_target": 0.15,
        "hybrid_q_frac_target": 0.80,
        "hybrid_pr_qc_threshold": 2e-3,
        "hybrid_pr_q_threshold": 8e-3,
        "hybrid_q_start_epoch": 200,
        "hybrid_q_ramp_epochs": 200,
        "hybrid_vf_only_min_epochs": 50,
        "hybrid_vf_only_max_epochs": 400,
        "hybrid_vf_only_target_vf": 1e-3,
        "hybrid_geom_ramp_epochs": 100,
        "hybrid_stop_on_rel_err": False,
        "hybrid_rel_err_target": 1e-9,
        "hybrid_rel_err_min_epochs": 100,
        "hybrid_use_val_erel_threshold": True,
        "hybrid_stop_on_pert_ratio": True,
        "hybrid_pert_alpha": 10.0,
        "hybrid_pert_metric": "p95",
        "hybrid_pert_componentwise": False,
        "hybrid_pert_min_epochs": 100,
        "hybrid_num_r_bins": 8,
        "hybrid_r_bins_mode": "linear",
        "hybrid_log_r_binned_val": True,
        "hybrid_r_binned_top_k": 2,
        "hybrid_r_binned_sort_key": "cons",
    }
    experiment = str(training_params.get("experiment", "blackbox")).lower()
    if experiment == "blackbox":
        model_params = {
            "key": key,
            "model_class": BlackBoxDHNN,
            "hidden_dim": 64,
            "h_scale": 1.0,
            "d_scale": 0.5,
        }
    elif experiment == "hybrid_eob":
        model_params = {
            "key": key,
            "model_class": Hybrid_EOB_DHNN,
            "hidden_dim_A": 32,
            "hidden_dim_D": 32,
            "hidden_dim_Q": 32,
            "hidden_dim_f": 32,
            "output_init_scale_A": 1.,
            "output_init_scale_D": 1.,
            "output_init_scale_Q": 1.,
            "output_init_scale_f": 1.,
            "A_floor": 1e-4,
            "D_floor": 1e-4,
            "Q_floor": 0.0,
            "f_floor": 1e-4,
            "A_max": 4.0,
            "D_max": 4.0,
            "Q_max": 8.0,
            "f_max": 8.0,
        }
    elif experiment == "eob_v2":
        model_params = {
            "key": key,
            "model_class": Neural_EOB_V2,
            "enable_A": True,
            "enable_D": True,
            "enable_Q": False,
            "enable_f": True,
            "enable_delta": False,
            "basis_order_A": 3,
            "basis_order_D": 3,
            "basis_order_Q": 3,
            "basis_order_f": 3,
            "basis_order_delta": 3,
            "hidden_dim_A": 32,
            "hidden_dim_D": 32,
            "hidden_dim_Q": 32,
            "hidden_dim_f": 32,
            "hidden_dim_delta": 32,
            "output_init_scale_A": 5e-3,
            "output_init_scale_D": 5e-3,
            "output_init_scale_Q": 5e-3,
            "output_init_scale_f": 5e-3,
            "output_init_scale_delta": 5e-3,
            "A_corr_bound": 0.5,
            "D_corr_bound": 0.1,
            "Q_corr_bound": 0.2,
            "f_corr_bound": 0.1,
            "delta_corr_bound": 0.1,
        }
    else:
        raise ValueError(
            f"Unknown training_params['experiment']={experiment!r}. "
            "Use 'blackbox', 'hybrid_eob', or 'eob_v2'."
        )
    # load training data
    x_train = jnp.load("seob_x_train_prelim.npy")
    y_train = jnp.load("seob_y_train_prelim.npy")
    x_val = jnp.load("seob_x_val_prelim.npy")
    y_val = jnp.load("seob_y_val_prelim.npy")
    val_data = (x_val, y_val)
    val_erel_path = Path("seob_erel_val_prelim.npy")
    if val_erel_path.exists():
        e_rel_val = jnp.load(str(val_erel_path))
        val_data = (x_val, y_val, e_rel_val)
        print(f"Loaded validation perturbation threshold data from: {val_erel_path}")
    else:
        print("WARNING: seob_erel_val_prelim.npy not found; perturbation-threshold stopping disabled.")
    # train model
    if experiment == "blackbox":
        trained_model = train_blackbox_dhnn_model_prelim(
            (x_train, y_train), val_data, model_params, training_params
        )
    elif experiment == "hybrid_eob":
        trained_model = train_hybrid_eob_dhnn_model_prelim(
            (x_train, y_train), val_data, model_params, training_params
        )
    else:
        trained_model = train_dhnn_model_prelim(
            (x_train, y_train), val_data, model_params, training_params
        )
