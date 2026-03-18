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

def _component_rms(y, eps=1e-8):
    """_component_rms
    
    Per-output RMS magnitude used for zero-centered residual normalization.
    
    Args:
        y (jnp.ndarray): Batch of data, shape `(N, C)`.
        eps (float): Minimum RMS magnitude.
        
    Returns:
        jnp.ndarray: Component-wise RMS, shape `(C,)`.
    """
    # Standardize by energy/magnitude rather than centered spread
    return jnp.maximum(jnp.sqrt(jnp.mean(y**2, axis=0)), eps)

def _validate_rhs_targets(y: jnp.ndarray, split_name: str, strict: bool = True, tol: float = 1e-14):
    """_validate_rhs_targets
    
    Detect obvious target-layout issues before expensive training starts.
    
    Ensures input targets comply with this dataset's explicit structural assumption
    of [dr/dt, dphi/dt, dp_rstar/dt, dp_phi/dt].
    
    Args:
        y (jnp.ndarray): Output targets.
        split_name (str): Label for errors.
        strict (bool): Should anomalous datasets trigger an exception or just a warning?
        tol (float): Tolerance for detecting structural duplicates across columns.
    """
    if y.ndim != 2 or y.shape[1] != 4:
        msg = f"{split_name}: expected RHS shape (N, 4), got {tuple(y.shape)}."
        raise ValueError(msg)

    duplicate_pairs = []
    # Exhaustively search for duplicate channels indicating flawed extraction.
    for i in range(4):
        for j in range(i + 1, 4):
            max_abs_diff = jnp.max(jnp.abs(y[:, i] - y[:, j]))
            if float(max_abs_diff) <= tol:
                duplicate_pairs.append((i, j))

    # Expect mostly positive dphi/dt if channel 1 is correctly assigned.
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

def _componentwise_relative_error_metrics_from_pred(y_pred, y_true, eps=1e-12):
    """_componentwise_relative_error_metrics_from_pred
    
    Componentwise relative error metrics.

    Relative error per component is defined as:
        abs((y_pred_i - y_true_i) / y_true_i)
    with denominator regularized by `eps` for numerical stability.

    Args:
        y_pred (jnp.ndarray): Predicted RHS batch.
        y_true (jnp.ndarray): Target RHS batch.
        eps (float): Denominator regularization.
    
    Returns:
        tuple: (Global relative error mean, Component-wise relative error mean array)
    """
    rel_abs = jnp.abs(y_pred - y_true) / (jnp.abs(y_true) + eps)
    rel_abs_mean = jnp.mean(rel_abs)
    rel_abs_comp = jnp.mean(rel_abs, axis=0)
    return rel_abs_mean, rel_abs_comp


def _relative_error_matrix_from_pred(y_pred, y_true, eps=1e-12):
    """_relative_error_matrix_from_pred
    
    Per-sample, per-component relative error matrix.
    
    Args:
        y_pred (jnp.ndarray): Predicted RHS.
        y_true (jnp.ndarray): Target RHS.
        eps (float): Denominator regularization.
        
    Returns:
        jnp.ndarray: Matrix of absolute relative errors.
    """
    return jnp.abs((y_pred - y_true) / (jnp.abs(y_true) + eps))


def _relative_error_summary(rel: Union[jnp.ndarray, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
    """_relative_error_summary
    
    Robust summary for relative-error arrays with finite filtering.
    
    Aggregates a matrix of relative errors into mean and 95th-percentile (p95) overall,
    and also breaks them out per component, effectively ignoring NaN/Inf outliers which 
    can frequently occur when ground truth is nearly zero.
    
    Args:
        rel (Union[jnp.ndarray, np.ndarray]): 2D array of relative errors.
        
    Returns:
        Dict: Contains 'mean', 'p95', 'comp_mean', 'comp_p95', and 'finite_ratio'.
    """
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
    """_split_supervision_data
    
    Accept (x, y) or (x, y, e_rel_ref) structured data.
    
    Helper function to safely unpack training or validation tuples that might natively include
    an extra reference array for relative-error stopping conditions.
    
    Args:
        data (tuple or list): Validation or train data tuple.
        split_name (str): The dataset's name (for descriptive errors).
        
    Returns:
        tuple: Extracted `x, y, e_rel_ref` matching the signature where `e_rel_ref` defaults to None.
    """
    if not isinstance(data, (tuple, list)):
        raise TypeError(f"{split_name} data must be tuple/list, got {type(data)}.")
    if len(data) < 2:
        raise ValueError(f"{split_name} data must have at least (x, y).")
    x, y = data[0], data[1]
    e_rel_ref = data[2] if len(data) >= 3 else None
    return x, y, e_rel_ref


def save_model_weights(model, path: str):
    """save_model_weights
    
    Save model parameters for future reloading.
    
    Args:
        model (eqx.Module): The current model.
        path (str): Filepath to serialize Equinox leaves to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(out_path), model)
    print(f"Saved model weights to: {out_path}")


def load_model_weights(model, path: str, strict: bool = True):
    """load_model_weights
    
    Load model parameters into an existing model structure.
    
    Args:
        model (eqx.Module): Base module structure to load parameters into.
        path (str): Filepath to deserialize from.
        strict (bool): Raise an exception if the file isn't found instead of warning.
        
    Returns:
        eqx.Module: The loaded model state.
    """
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

def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """_masked_mean
    
    Mean over masked samples (returns 0 when mask is empty).
    
    Args:
        values (jnp.ndarray): An array to be averaged.
        mask (jnp.ndarray): Binary mask outlining indices to incorporate.
        
    Returns:
        jnp.ndarray: Average scalar, zeroed if the entire mask is empty.
    """
    mask_f = mask.astype(values.dtype)
    denom = jnp.maximum(jnp.sum(mask_f), 1.0)
    return jnp.sum(values * mask_f) / denom


def _safe_log_abs(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """_safe_log_abs
    
    Stable log(|x|).
    
    Args:
        x (jnp.ndarray): Array of potentially small or zero values.
        eps (float): Float floor to truncate argument.
        
    Returns:
        jnp.ndarray: Bounded equivalent of `log(abs(x))`.
    """
    return jnp.log(jnp.maximum(jnp.abs(x), eps))

def train_hybrid_eob_dhnn_model_prelim(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    val_data: Tuple[jnp.ndarray, jnp.ndarray],
    model_params: Dict[str, int],
    training_params: Dict[str, Union[int, float]],
) -> Hybrid_EOB_DHNN:
    """train_hybrid_eob_dhnn_model_prelim

    Train the Hybrid EOB DHNN with multi-objective potential-aware loss.

    Accepts a flat configuration dictionary with the following keys:

    Core training:
        adam_epochs (int): Number of total training epochs.
        batch_size (int): Mini-batch size per gradient step.
        learning_rate (float): Peak Adam learning rate (default 3e-4).
        warmup_steps (int): LR warm-up steps (default 10% of total steps).
        loss_eps (float): Denominator regularization floor (default 1e-8).
        ema_alpha (float): EMA smoothing factor for adaptive weights (default 0.15).

    Curriculum:
        curriculum_target_vf (float): VF loss threshold to unlock geometry channels (default 1e-3).
        curriculum_min_epochs (int): Minimum epochs before geometry unlock (default 50).
        curriculum_max_epochs (int): Maximum epochs before forcing geometry unlock (default 400).
        curriculum_ramp_epochs (int): Epochs to ramp geometry + Q gains from 0 to 1 (default 100).

    pr-channel masking (always adaptive):
        qc_frac (float): Quantile defining quasi-circular pr threshold (default 0.15).
        q_frac (float): Quantile defining non-circular pr lower bound (default 0.80).

    Static loss weights (multiplied by EMA-dynamic inverse):
        w_flux (float): Flux channel weight (default 1.0).
        w_omega (float): Omega channel weight (default 1.0).
        w_cons (float): Conservative channel weight (default 1.0).
        w_q (float): Q channel weight (default 0.5).

    Perturbation-ratio monitoring:
        pert_alpha (float): Target Val/pert p95 ratio (default 10.0).
        stop_on_pert_ratio (bool): Stop training when ratio <= pert_alpha (default False).
        pert_min_epochs (int): Minimum epochs before pert-ratio stopping is active (default 100).

    Weight I/O:
        load_weights_path (str): Path to load weights from (default empty = skip).
        save_weights_path (str): Path to save weights to (default 'hybrid_eob_weights.eqx').

    Diagnostics:
        log_r_binned_val (bool): Log per-r-bin validation breakdown every 10 epochs (default True).
    """
    # --- Initialize the model ---
    
    model_class = model_params.get("model_class", Hybrid_EOB_DHNN)
    model_kwargs = {k: v for k, v in model_params.items() if k != "model_class"}
    key = model_kwargs["key"]
    model = model_class(**model_kwargs)
    load_weights_path = str(training_params.get("load_weights_path", "")).strip()
    save_weights_path = str(training_params.get("save_weights_path", "hybrid_eob_weights.eqx")).strip()
    if load_weights_path:
        model = load_model_weights(model, load_weights_path, strict=True)

    # --- Load training and validation datasets ---

    x_train, y_train, _ = _split_supervision_data(train_data, "train")
    x_val, y_val, e_rel_val_ref = _split_supervision_data(val_data, "val")
    batch_size = int(training_params["batch_size"])
    _validate_rhs_targets(y_train, "train", strict=True)
    _validate_rhs_targets(y_val, "val", strict=True)

    num_train_samples = x_train.shape[0]
    effective_batch_size = min(batch_size, num_train_samples)
    num_train_batches = max(1, num_train_samples // effective_batch_size)
    used_train_samples = num_train_batches * effective_batch_size
    dropped_train_samples = num_train_samples - used_train_samples

    # --- Initialize the optimizer ---

    lr_peak = float(training_params.get("learning_rate", 3e-4))
    lr_init = 0.1 * lr_peak
    lr_end = 0.01 * lr_peak
    total_steps = int(training_params["max_epochs"]) * num_train_batches
    warmup_steps_requested = int(0.1 * total_steps)
    warmup_steps = min(max(0, warmup_steps_requested), max(0, total_steps - 1))
    decay_steps = max(total_steps, warmup_steps + 1)
    single_stage_lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_init,
        peak_value=lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_end,
    )
    lr_schedule = optax.join_schedules(
        [single_stage_lr_schedule] * 3, 
        [total_steps, total_steps]       
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

    # --- Perturbation-ratio monitoring
    use_val_erel_threshold = True
    stop_on_pert_ratio = bool(training_params.get("stop_on_pert_ratio", False))
    pert_alpha = float(training_params.get("pert_alpha", 10.0))
    pert_metric = "p95"
    pert_componentwise = False
    pert_min_epochs = int(training_params.get("pert_min_epochs", 100))

    # --- Diagnostics: r-binned validation breakdown
    log_r_binned_val = bool(training_params.get("log_r_binned_val", True))
    num_r_bins = 8
    r_bins_mode = "linear"
    r_binned_top_k = 2
    r_binned_sort_key = "cons"

    # --- No relative-error early stopping (deprecated path)
    stop_on_rel_err = False
    rel_err_target = 1e-9
    rel_err_min_epochs = pert_min_epochs

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
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def get_trainable_mask(model, s0_g, s1_g, s2_g, s3_g):
        """get_trainable_mask
        
        Generate a boolean mask tree for selectively freezing neural heads.
        - Stage 0: Conservative coordinate only (A, D, Q trainable, f frozen)
        - Stage 1: Dissipative only (f trainable, A, D, Q frozen)
        - Stage 2: Conservative radial momentum (Q trainable, A, D, f frozen)
        - Stage 3: Fine-tuning (All heads trainable)
        """
        # Start with all frozen structure
        mask = jax.tree_util.tree_map(lambda _: False, model)
        is_inexact = jax.tree_util.tree_map(eqx.is_inexact_array, model)

        if float(s3_g) >= 1.0:
            # Stage 3: Fine tuning (All)
            mask = jax.tree_util.tree_map(lambda _: True, model)
        elif float(s2_g) >= 1.0:
            # Stage 2: Conservative radial momentum (Q_head)
            # Use tree_map to ensure the mask sub-tree matches the head structure
            Q_mask = jax.tree_util.tree_map(lambda _: True, model.Q_head)
            mask = eqx.tree_at(lambda m: m.Q_head, mask, Q_mask)
        elif float(s1_g) >= 1.0:
            # Stage 1: Dissipative only (f_head)
            # Use tree_map to ensure the mask sub-tree matches the head structure
            f_mask = jax.tree_util.tree_map(lambda _: True, model.f_head)
            mask = eqx.tree_at(lambda m: m.f_head, mask, f_mask)
        else:
            # Stage 0: Conservative only (A, D, Q)
            # Use tree_map to ensure structural matching for all selected heads
            A_mask = jax.tree_util.tree_map(lambda _: True, model.A_head)
            D_mask = jax.tree_util.tree_map(lambda _: True, model.D_head)
            Q_mask = jax.tree_util.tree_map(lambda _: True, model.Q_head)
            mask = eqx.tree_at(lambda m: (m.A_head, m.D_head, m.Q_head), mask, (A_mask, D_mask, Q_mask))
        
        # We must return None where is_inexact is False to match the structural presence of None in grads
        return jax.tree_util.tree_map(lambda m, i: m if i else None, mask, is_inexact)

    def hybrid_loss_terms(m, x, y, stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain):
        """hybrid_loss_terms
        
        Evaluate the multi-objective loss for the Hybrid EOB model.
        
        Each geometric loss term has its own gain scalar so that they can be
        activated sequentially as the vector field loss converges:
        - l_omega (A potential via QC dispersion) activates first
        - l_q (strong-field D/Q sector) activates along with l_omega
        - l_flux (f network, needs correct omega input) unlocks after omega converges
        - l_cons (conservative flow in radial sector) unlocks last
        
        Args:
            m (eqx.Module): Evaluated Model.
            x (jnp.ndarray): Input batch.
            y (jnp.ndarray): Target RHS.
            stage_0_gain (float): Ramp weight for conservative only training.
            stage_1_gain (float): Ramp weight for flux training.
            stage_2_gain (float): Ramp weight for conservative radial momentum training.
            stage_3_gain (float): Ramp weight for fine-tuning.
            
        Returns:
            Tuple containing total combined loss and auxiliary unpackable individual losses.
        """

        eps = float(training_params.get("loss_eps", 1e-8))
        # Stage 3 loss is plain vector field
        y_pred = m(x)
        rel_sq = ((y_pred - y) / (jnp.abs(y) + eps)) ** 2  # (N, 4)
        l_vf = jnp.mean(rel_sq)
        l_stage_3 = l_vf
        rel_abs_mean, rel_abs_comp = _componentwise_relative_error_metrics_from_pred(y_pred, y, eps=eps)
        
        p_rstar = x[:, 3]
        p_phi = x[:, 4]
        p_rstar_safe = jnp.where(
            jnp.abs(p_rstar) < 1e-12,
            jnp.where(p_rstar >= 0.0, 1e-12, -1e-12),
            p_rstar,
        )
        p_phi_safe = jnp.where(
            jnp.abs(p_phi) < 1e-12,
            jnp.where(p_phi >= 0.0, 1e-12, -1e-12),
            p_phi,
        )
        
        # Stage 0 loss is purely conservative channels
        omega_true = jnp.maximum(y[:, 1], eps)
        omega_pred = jnp.maximum(y_pred[:, 1], eps)
        dr_ratio_true = y[:, 0] / p_rstar_safe
        dr_ratio_pred = y_pred[:, 0] / p_rstar_safe
        l_q = jnp.mean(((dr_ratio_pred - dr_ratio_true) / (jnp.abs(dr_ratio_true) + 1e-12)) ** 2)
        omega_err_sq = ((omega_pred - omega_true) / (jnp.abs(omega_true) + 1e-12)) ** 2
        l_omega = jnp.mean(omega_err_sq)
        l_stage_0 = l_omega + l_q

        # Stage 1 Flux
        flux_true = y[:, 3]
        flux_pred = y_pred[:, 3]
        l_flux = jnp.mean(((flux_pred - flux_true) / (jnp.abs(flux_true) + 1e-12)) ** 2)
        l_stage_1 = l_flux
        
        # Stage 2 Conservative radial momentum
        cons_true = y[:, 2] - flux_true * (p_rstar / p_phi_safe)
        cons_pred = y_pred[:, 2] - flux_pred * (p_rstar / p_phi_safe)
        l_cons = jnp.mean(((cons_pred - cons_true) / (jnp.abs(cons_true) + 1e-12)) ** 2)
        l_stage_2 = l_cons
        
        l_total = (
            stage_0_gain * l_stage_0
            + stage_1_gain * l_stage_1
            + stage_2_gain * l_stage_2
            + stage_3_gain * l_stage_3
        )
        return l_total, (l_vf, l_flux, l_omega, l_cons, l_q, rel_abs_mean, rel_abs_comp)

    @eqx.filter_jit
    def step(diff_model, static_model, opt_state, x, y, stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain, trainable_mask):
        """step
        
        Single optimization step updating model configuration towards multi-objective loss.
        Selective head freezing is applied by zeroing out gradients and updates for frozen heads.
        """
        def loss_fn(m):
            return hybrid_loss_terms(m, x, y, stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain)

        (loss_value, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(diff_model)
        
        # Zero-out gradients for frozen heads
        # Gracefully handle None leaves to match JAX structural expectations
        grads = jax.tree_util.tree_map(lambda g, m: jnp.where(m, g, 0.0) if g is not None else None, grads, trainable_mask)
        
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        
        # Zero-out updates for frozen heads (prevents momentum drift)
        updates = jax.tree_util.tree_map(lambda u, m: jnp.where(m, u, 0.0) if u is not None else None, updates, trainable_mask)
        
        diff_model = eqx.apply_updates(diff_model, updates)
        l_vf, l_flux, l_omega, l_cons, l_q, rel_abs_mean, rel_abs_comp = aux
        return (
            diff_model,
            opt_state,
            loss_value,
            l_vf,
            l_flux,
            l_omega,
            l_cons,
            l_q,
            rel_abs_mean,
            rel_abs_comp,
        )

    def finite_output_ratio(model, x):
        """finite_output_ratio
        
        Returns ratio of valid/finite network outputs across an evaluation batch."""
        y_pred = model(x)
        return jnp.mean(jnp.isfinite(y_pred))

    def r_binned_val_metrics(model, x_val, y_val, stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain):
        """r_binned_val_metrics
        
        Discretize validation responses across radial separation distance (`r`) bins.
        
        Helps isolate areas in phase-space where structural breakdowns occur (e.g. merger).
        
        Args:
            model (eqx.Module): Model predicting the properties.
            x_val (jnp.ndarray): Validation input space.
            y_val (jnp.ndarray): Validation target targets.
            stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain (float): Per-stage curriculum gain scalars.
            
        Returns:
            list: Dictionary table denoting metrics calculated solely inside bounds of `r_bin_edges`.
        """
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
            _, aux_bin = hybrid_loss_terms(model, x_bin, y_bin, stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain)
            l_vf_b, l_flux_b, l_omega_b, l_cons_b, l_q_b, *_ = aux_bin
            rows.append(
                {
                    "bin": b,
                    "r_lo": lo,
                    "r_hi": hi,
                    "n": count,
                    "vf": float(l_vf_b),
                    "flux": float(l_flux_b),
                    "omega": float(l_omega_b),
                    "cons": float(l_cons_b),
                    "q": float(l_q_b),
                }
            )
        return rows

    def compact_r_binned_summary(rows):
        """compact_r_binned_summary
        
        Compile a compact presentation layer tracking the highest-loss radial boundaries."""
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

    # --- Stage unlock parameters
    # Stage 0: pure VF training — runs for stage0_epochs (force-unlocks stage 1 at this epoch).
    # Stage 1: conservative channels (omega+rdot) — activates after stage 0.
    # Stage 2: dissipative channels (flux+cons_diss) — threshold-only, never force-unlocked.
    any_stage_min_epochs  = int(training_params.get("min_epochs",  50))
    any_stage_max_epochs  = int(training_params.get("max_epochs",  2000))
    
    # --- Per-term unlock state ---
    # Each geometric term has its own unlock epoch and ramp, activated sequentially:
    #   0. vf (vector field) — unlocks first
    #   1. omega + rdot (A potential via QC dispersion + radial velocity) — unlocks after vf converges
    #   2. flux  + prdot (f network + radial momentum) — unlocks after omega converges
    stage_0_unlock_epoch = 0
    stage_1_unlock_epoch = -1
    stage_2_unlock_epoch = -1
    stage_3_unlock_epoch = -1

    stage_0_gain_now = 1.0
    stage_1_gain_now = 0.0
    stage_2_gain_now = 0.0
    stage_3_gain_now = 0.0
    
    # VF thresholds that trigger each stage unlock.
    # Stage 1 (omega+rdot): force-unlocks at stage0_epochs; also unlocks early if VF ≤ stage0_vf_target.
    # Stage 2 (flux, cons_diss): threshold-only — only unlock if VF actually improves.
    loss_target_0 = float(training_params.get("stage0_loss_target", 1e-4))
    loss_target_1  = float(training_params.get("stage1_loss_target",1e-5))
    loss_target_2  = float(training_params.get("stage2_loss_target",1e-6))
    loss_target_3  = float(training_params.get("stage3_loss_target",1e-7))
    
    print("HybridEOB per-term unlock thresholds:",
          {"stage_0": loss_target_0, "stage_1": loss_target_1,
           "stage_2": loss_target_2, "stage_3": loss_target_3})

    last_val_loss = jnp.array(jnp.inf, dtype=x_train.dtype)
    last_val_rel_summary = None
    last_val_pert_ratio = np.inf
    last_val_pert_ratio_comp = None

    @eqx.filter_jit
    def scan_epoch(diff_model, static_model, opt_state, batch_indices_in,
                   stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain, trainable_mask):
        def scan_step(carry, batch_idx):
            dm, opt = carry
            x_batch = jnp.take(x_train, batch_idx, axis=0)
            y_batch = jnp.take(y_train, batch_idx, axis=0)
            (
                dm_next,
                opt_next,
                b_loss,
                b_l_vf,
                b_l_flux,
                b_l_omega,
                b_l_cons,
                b_l_q,
                b_rel_abs_mean,
                b_rel_abs_comp,
            ) = step(dm, static_model, opt, x_batch, y_batch, stage_0_gain, stage_1_gain, stage_2_gain, stage_3_gain, trainable_mask)
            metrics = jnp.stack([b_loss, b_l_vf, b_l_flux, b_l_omega, b_l_cons, b_l_q, b_rel_abs_mean])
            return (dm_next, opt_next), (metrics, b_rel_abs_comp)

        (diff_model_out, opt_state_out), (all_metrics, all_comp) = jax.lax.scan(
            scan_step, (diff_model, opt_state), batch_indices_in)
        mean_metrics = jnp.mean(all_metrics, axis=0)
        mean_comp = jnp.mean(all_comp, axis=0)
        return diff_model_out, static_model, opt_state_out, mean_metrics, mean_comp

    for epoch in range(int(training_params["adam_epochs"])):
        key, key_train = jax.random.split(key, 2)
        perm = jax.random.permutation(key_train, num_train_samples)
        batch_indices = perm[:used_train_samples].reshape((num_train_batches, effective_batch_size))

        loss_now = float(last_val_loss)

        # Sequential unlock: each term fires only after its VF threshold is met
        # (or the hard max-epoch cap is hit) AND after the minimum epoch.
        if (stage_1_unlock_epoch < 0) and (epoch >= any_stage_min_epochs):
            if (loss_now <= loss_target_0) or (epoch >= any_stage_max_epochs):
                stage_0_gain_now = 0.1
                stage_1_gain_now = 1.0
                stage_1_unlock_epoch = epoch
                save_model_weights(model, f"{save_weights_path}_stage_0.eqx")
                print(f"[HybridEOB] Stage 1 unlocked at epoch {epoch} (val_loss={loss_now:.4g})")

        if (stage_2_unlock_epoch < 0) and (stage_1_unlock_epoch > 0) and (epoch >= stage_1_unlock_epoch + any_stage_min_epochs):
            if (loss_now <= loss_target_1) or (epoch >= stage_1_unlock_epoch + any_stage_max_epochs):
                stage_2_unlock_epoch = epoch
                stage_1_gain_now = 0.1
                stage_2_gain_now = 1.0
                save_model_weights(model, f"{save_weights_path}_stage_1.eqx")
                print(f"[HybridEOB] Stage 2 unlocked at epoch {epoch} (val_loss={loss_now:.4g})")
        
        if (stage_3_unlock_epoch < 0) and (stage_2_unlock_epoch > 0) and (epoch >= stage_2_unlock_epoch + any_stage_min_epochs):
            if (loss_now <= loss_target_2) or (epoch >= stage_2_unlock_epoch + any_stage_max_epochs):
                stage_3_unlock_epoch = epoch
                stage_2_gain_now = 0.1
                stage_3_gain_now = 1.0
                save_model_weights(model, f"{save_weights_path}_stage_2.eqx")
                

        stage0_g_arr = jnp.array(stage_0_gain_now, dtype=x_train.dtype)
        stage1_g_arr  = jnp.array(stage_1_gain_now,  dtype=x_train.dtype)
        stage2_g_arr  = jnp.array(stage_2_gain_now,  dtype=x_train.dtype)
        stage3_g_arr  = jnp.array(stage_3_gain_now,  dtype=x_train.dtype)

        diff_model, static_model = eqx.partition(model, eqx.is_inexact_array)
        trainable_mask = get_trainable_mask(model, stage0_g_arr, stage1_g_arr, stage2_g_arr, stage3_g_arr)
        
        diff_model, static_model, opt_state, epoch_metrics, train_rel_abs_comp = scan_epoch(
            diff_model, static_model, opt_state, batch_indices,
            stage0_g_arr, stage1_g_arr, stage2_g_arr, stage3_g_arr, trainable_mask
        )
        model = eqx.combine(diff_model, static_model)
        train_loss = epoch_metrics[0]
        train_l_vf = epoch_metrics[1]
        train_l_flux = epoch_metrics[2]
        train_l_omega = epoch_metrics[3]
        train_l_cons = epoch_metrics[4]
        train_l_q = epoch_metrics[5]
        train_rel_abs_mean = epoch_metrics[6]

        val_loss, val_aux = hybrid_loss_terms(model, x_val, y_val, stage_0_gain_now, stage_1_gain_now, stage_2_gain_now, stage_3_gain_now)
        (
            val_l_vf,
            val_l_flux,
            val_l_omega,
            val_l_cons,
            val_l_q,
            val_rel_abs_mean,
            val_rel_abs_comp,
        ) = val_aux
        last_val_l_vf = val_l_vf

        # Validation-only threshold metric: compare NN relative error to SEOB perturbation floor.
        if (epoch % 10 == 0) or (epoch >= rel_err_min_epochs):
            y_val_pred_for_rel = model(x_val)
            val_rel_matrix = _relative_error_matrix_from_pred(y_val_pred_for_rel, y_val, eps=1e-8)
            last_val_rel_summary = _relative_error_summary(val_rel_matrix)
            if pert_ref_summary is not None:
                if pert_metric == "p95":
                    denom = max(float(pert_ref_summary["p95"]), 1e-8)
                    numer = float(last_val_rel_summary["p95"])
                    denom_comp = np.maximum(np.asarray(pert_ref_summary["comp_p95"]), 1e-8)
                    numer_comp = np.asarray(last_val_rel_summary["comp_p95"])
                else:
                    denom = max(float(pert_ref_summary["mean"]), 1e-8)
                    numer = float(last_val_rel_summary["mean"])
                    denom_comp = np.maximum(np.asarray(pert_ref_summary["comp_mean"]), 1e-8)
                    numer_comp = np.asarray(last_val_rel_summary["comp_mean"])
                last_val_pert_ratio = numer / denom
                last_val_pert_ratio_comp = numer_comp / denom_comp

        if epoch % 10 == 0:
            val_finite_ratio = finite_output_ratio(model, x_val)
            print(
                f"[HybridEOB] Epoch {epoch}, Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}, "
                f"Val VF: {val_l_vf:.3e}, Val Flux: {val_l_flux:.3e}, Val Omega: {val_l_omega:.3e}, "
                f"Val Cons: {val_l_cons:.3e}, Val Q: {val_l_q:.3e}, "
                f"ValRel(mean,p95)=({last_val_rel_summary['mean']:.3e}, {last_val_rel_summary['p95']:.3e}), "
                f"Val finite ratio: {val_finite_ratio:.3e}"
            )
            if pert_ref_summary is not None:
                print(
                    f"Val/SEOB-pert ratio ({pert_metric}): "
                    f"{last_val_pert_ratio:.3e} (target <= {pert_alpha:.3e})"
                )
                print(
                    "Val/SEOB-pert component ratios:",
                    np.asarray(last_val_pert_ratio_comp),
                )
            print(
                "RelAbsComp(train,val):",
                np.asarray(train_rel_abs_comp),
                np.asarray(val_rel_abs_comp),
            )
            if log_r_binned_val:
                binned_rows = r_binned_val_metrics(model, x_val, y_val, stage_0_gain_now, stage_1_gain_now, stage_2_gain_now, stage_3_gain_now)
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
        save_model_weights(model, f"{save_weights_path}_final.eqx")
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # training parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    experiment = "hybrid_eob"
    hidden_dim = 32
    p_dim = 4
    q_dim = 5
    training_params = {
        "experiment": "hybrid_eob",
        # -- Core --
        "learning_rate": 3e-4,
        "batch_size": 8192,
        "loss_eps": 1e-8,
        # -- Weight I/O --
        "load_weights_path": "",
        "save_weights_path": f"saved_models/{experiment}_weights_{hidden_dim}_{p_dim}_{q_dim}",
        # --- Training stages ---
        # Stage 0: pure VF loss for ~2000 epochs
        # Stage 1: conservative channels (omega=dphi/dt, rdot=dr/dt) fire together.
        # Stage 2: dissipative channels (flux+prdot) fire together.
        # adam_epochs covers stage 0 + stage 1 ramp + stage 2 ramp + 500-epoch buffer.
        "min_epochs":          50,   # Never unlock before this many epochs
        "max_epochs":       1000,    # Move to next stage after this many epochs
        "adam_epochs":       3000,    # Total epochs for adam = max_epochs * num of stages
        # -- pr-mask quantiles --
        "qc_frac": 0.15,
        "q_frac": 0.80,
        # -- Pert-ratio monitoring --
        "stop_on_pert_ratio": True,
        "pert_alpha": 10.0,
        "pert_min_epochs": 100,
        # -- Diagnostics --
        "log_r_binned_val": True,
    }
    model_params = {
        "key": key,
        "model_class": Hybrid_EOB_DHNN,
        "hidden_dim_A": hidden_dim,
        "hidden_dim_D": hidden_dim,
        "hidden_dim_Q": hidden_dim,
        "hidden_dim_f": hidden_dim,
        # Padé order for the rational activation: P[degree_of_p] / Q[degree_of_q]
        # per neuron per head. P[4]/Q[5] gives 9 free parameters per neuron.
        "degree_of_p": p_dim,
        "degree_of_q": q_dim,
        "A_floor": 1e-4,
        "D_floor": 1e-4,
        "Q_floor": 1e-4,
        "f_floor": 1e-4,
        "A_max": 8.0,
        "D_max": 8.0,
        "Q_max": 8.0,
        "f_max": 8.0,
    }
    # load training data
    x_train = np.load("seob_x_train_prelim.npy")
    y_train = np.load("seob_y_train_prelim.npy")
    x_val = np.load("seob_x_val_prelim.npy")
    y_val = np.load("seob_y_val_prelim.npy")
    val_data = (x_val, y_val)
    val_erel_path = Path("seob_erel_val_prelim.npy")
    if val_erel_path.exists():
        e_rel_val = jnp.load(str(val_erel_path))
        val_data = (x_val, y_val, e_rel_val)
        print(f"Loaded validation perturbation threshold data from: {val_erel_path}")
    else:
        print("WARNING: seob_erel_val_prelim.npy not found; perturbation-threshold stopping disabled.")
    # train model
    trained_model = train_hybrid_eob_dhnn_model_prelim(
        (x_train, y_train), val_data, model_params, training_params
    )