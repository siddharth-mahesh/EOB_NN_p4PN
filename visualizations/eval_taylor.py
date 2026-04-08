import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import sys
import os

# Add paths for project imports
PROJECT_ROOT = "/home/sidmahesh/EOB_NN_p4PN"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "EOB_NN_p4PN"))

from EOB_NNp4PN.eob_nnp4pn_training_module_hybrid import Hybrid_EOB_DHNN
from EOB.eob_constants_3pn import set_eob_constants_3PN

def e_pc(th,pred):
    th_safe = jnp.where(jnp.abs(th)<1e-10,1.,th)
    return 100*jnp.abs(pred-th_safe)/jnp.abs(th_safe)

def main():
    # 1. Initialize Network Architecture (Depth 4, Hidden Dim 32)
    key = jax.random.PRNGKey(0)
    model = Hybrid_EOB_DHNN(
        key=key, 
        hidden_dim_A=32, hidden_dim_D=32, hidden_dim_Q=32, hidden_dim_f=32,
        degree_of_p=4, degree_of_q=5
    )

    # 2. Load the specific weights
    weight_path = os.path.join(PROJECT_ROOT, "saved_models/hybrid_eob_weights_32_4_5_stage_1.eqx")
    try:
        model = eqx.tree_deserialise_leaves(weight_path, model)
        print(f"--- Loaded weights: {weight_path} ---\n")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    # 3. Mass ratios for evaluation
    nus = jnp.linspace(0.0, 0.25, 100)
    nu_train_min_max = [10./121.,1./4.]
    u0 = jnp.array(0.0)
    A_0 = jnp.zeros_like(nus)
    e_A_0 = jnp.zeros_like(nus)
    A_1 = jnp.zeros_like(nus)
    e_A_1 = jnp.zeros_like(nus)
    A_2 = jnp.zeros_like(nus)
    e_A_2 = jnp.zeros_like(nus)
    A_3 = jnp.zeros_like(nus)
    e_A_3 = jnp.zeros_like(nus)
    A_4 = jnp.zeros_like(nus)
    e_A_4 = jnp.zeros_like(nus)
    D_0 = jnp.zeros_like(nus)
    e_D_0 = jnp.zeros_like(nus)
    D_1 = jnp.zeros_like(nus)
    e_D_1 = jnp.zeros_like(nus)
    D_2 = jnp.zeros_like(nus)
    e_D_2 = jnp.zeros_like(nus)
    D_3 = jnp.zeros_like(nus)
    e_D_3 = jnp.zeros_like(nus)
    Q_0 = jnp.zeros_like(nus)
    e_Q_0 = jnp.zeros_like(nus)
    f_0 = jnp.zeros_like(nus)
    e_f_0 = jnp.zeros_like(nus)
    f_1 = jnp.zeros_like(nus)
    e_f_1 = jnp.zeros_like(nus)
    f_2 = jnp.zeros_like(nus)
    e_f_2 = jnp.zeros_like(nus)
    f_3 = jnp.zeros_like(nus)
    e_f_3 = jnp.zeros_like(nus)
    f_3_l = jnp.zeros_like(nus)
    e_f_3_l = jnp.zeros_like(nus)

    for i , nu_val in enumerate(nus):
        nu = jnp.array(nu_val)
        cons = set_eob_constants_3PN(nu_val)
        
        print(f"=================================================================================")
        print(f"  Comparison for nu = {nu_val}")
        print(f"=================================================================================")

        # --- A POTENTIAL (1 + a1*u + a3*u^3 + a4*u^4) ---
        def nn_A(u):
            # Match A_0 to del A / del in_1 | in_1, in_2 = 0
            # in_1 = u_lin, in_2 = u_log
            raw = model.A_head(jnp.array([nu, u, 0.0]))
            return model._bounded_positive(raw, model.A_floor, model.A_max)
        
        # 3PN coefficients
        th_A = [1.0, cons["a_1"], 0.0, cons["a_3"], cons["a_4"]]
        
        # NN derivatives
        dA_0 = float(nn_A(u0))
        dA_1 = float(jax.jvp(nn_A, (u0,), (1.0,))[1])
        dA_2 = float(jax.jvp(lambda x: jax.jvp(nn_A, (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 2.0
        dA_3 = float(jax.jvp(lambda x: jax.jvp(lambda y: jax.jvp(nn_A, (y,), (1.0,))[1], (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 6.0
        dA_4 = float(jax.jvp(lambda x: jax.jvp(lambda y: jax.jvp(lambda z: jax.jvp(nn_A, (z,), (1.0,))[1], (y,), (1.0,))[1], (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 24.0
        
        A_0 = A_0.at[i].set(dA_0)
        e_A_0 = e_A_0.at[i].set(e_pc(th_A[0],dA_0))
        A_1 = A_1.at[i].set(dA_1)
        e_A_1 = e_A_1.at[i].set(e_pc(th_A[1],dA_1))
        A_2 = A_2.at[i].set(dA_2)
        e_A_2 = e_A_2.at[i].set(e_pc(th_A[2],dA_2))
        A_3 = A_3.at[i].set(dA_3)
        e_A_3 = e_A_3.at[i].set(e_pc(th_A[3],dA_3))
        A_4 = A_4.at[i].set(dA_4)
        e_A_4 = e_A_4.at[i].set(e_pc(th_A[4],dA_4))

        # --- D POTENTIAL (1 + d2*u^2 + d3*u^3) ---
        def nn_D(u):
            # Same matching logic for D (partial derivative wrt u_lin)
            raw = model.D_head(jnp.array([nu, u, 0.0]))
            return model._bounded_positive(raw, model.D_floor, model.D_max)

        th_D = [1.0, 0.0, cons["d_2"], cons["d_3"]]
        dD_0 = float(nn_D(u0))
        dD_1 = float(jax.jvp(nn_D, (u0,), (1.0,))[1])
        dD_2 = float(jax.jvp(lambda x: jax.jvp(nn_D, (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 2.0
        dD_3 = float(jax.jvp(lambda x: jax.jvp(lambda y: jax.jvp(nn_D, (y,), (1.0,))[1], (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 6.0

        D_0 = D_0.at[i].set(dD_0)
        e_D_0 = e_D_0.at[i].set(e_pc(th_D[0],dD_0))
        D_1 = D_1.at[i].set(dD_1)
        e_D_1 = e_D_1.at[i].set(e_pc(th_D[1],dD_1))
        D_2 = D_2.at[i].set(dD_2)
        e_D_2 = e_D_2.at[i].set(e_pc(th_D[2],dD_2))
        D_3 = D_3.at[i].set(dD_3)
        e_D_3 = e_D_3.at[i].set(e_pc(th_D[3],dD_3))

        # --- Q POTENTIAL (Constant z3) ---
        # Q head takes [nu, u, prstar]. Weak field limit is u->0, prstar->0.
        def nn_Q(u, pr):
            # Q takes [nu, u_lin, prstar, u_log]
            raw = model.Q_head(jnp.array([nu, u, pr, 0.0]))
            return model._bounded_positive(raw, model.Q_floor, model.Q_max)
        
        q_val = float(nn_Q(u0, 0.0))
        th_Q = cons["z_3"]

        Q_0 = Q_0.at[i].set(q_val)
        e_Q_0 = e_Q_0.at[i].set(e_pc(th_Q,q_val))

        # --- f POTENTIAL (1 + f1*x + f2*x^2 + f3*x^3) ---
        # f head takes [nu, x] where x = Omega^(2/3)
        def nn_f(x):
            # x is Omega^(1/3) in f_potential. Replicating partial wrt x_lin.
            raw = model.f_head(jnp.array([nu, x, 0.0]))
            return model._bounded_positive(raw, model.f_floor, model.f_max)
        
        th_f = [1.0, cons["f_1"], cons["f_2"], cons["f_3"], cons["f_3_l"]]
        df_0 = float(nn_f(u0))
        df_1 = float(jax.jvp(nn_f, (u0,), (1.0,))[1])
        df_2 = float(jax.jvp(lambda x: jax.jvp(nn_f, (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 2.0
        df_3 = float(jax.jvp(lambda x: jax.jvp(lambda y: jax.jvp(nn_f, (y,), (1.0,))[1], (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 6.0
        
        f_0 = f_0.at[i].set(df_0)
        e_f_0 = e_f_0.at[i].set(e_pc(th_f[0],df_0))
        f_1 = f_1.at[i].set(df_1)
        e_f_1 = e_f_1.at[i].set(e_pc(th_f[1],df_1))
        f_2 = f_2.at[i].set(df_2)
        e_f_2 = e_f_2.at[i].set(e_pc(th_f[2],df_2))
        f_3 = f_3.at[i].set(df_3)
        e_f_3 = e_f_3.at[i].set(e_pc(th_f[3],df_3))

        # --- f_3_l matching (log term) ---
        def nn_f_log_deriv(x_lin):
            f_at_x = lambda log_in: model._bounded_positive(model.f_head(jnp.array([nu, x_lin, log_in])), model.f_floor, model.f_max)
            return jax.grad(f_at_x)(0.0)

        df_3_l = float(jax.jvp(lambda x: jax.jvp(lambda y: jax.jvp(nn_f_log_deriv, (y,), (1.0,))[1], (x,), (1.0,))[1], (u0,), (1.0,))[1]) / 6.0
        f_3_l = f_3_l.at[i].set(df_3_l)
        e_f_3_l = e_f_3_l.at[i].set(e_pc(th_f[4],df_3_l))

    # --- PLOTTING ---
    print("\n--- Generating Plots ---")
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 200,
        'axes.titlesize': 14,
    })

    def apply_style(ax, title):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(r"Symmetric Mass Ratio $\nu$")
        ax.set_ylabel("Relative Error (%)")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        # Shade training range
        ax.axvspan(nu_train_min_max[0], nu_train_min_max[1], color='green', alpha=0.1, label='Training Region')
        ax.legend(loc='lower right', frameon=True, framealpha=0.5)
        # Use scientific notation for small values
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    # 1. Individual Plots
    print(" - Rendering individual potential plots...")
    
    # A Potential
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(nus, e_A_0, label=r"$a_0$ (const)", lw=2)
    ax.plot(nus, e_A_1, label=r"$a_1$ ($u$)", lw=2)
    ax.plot(nus, e_A_2, label=r"$a_2$ ($u^2$)", color='gray', linestyle='--', alpha=0.7)
    ax.plot(nus, e_A_3, label=r"$a_3$ ($u^3$)", lw=2)
    ax.plot(nus, e_A_4, label=r"$a_4$ ($u^4$)", lw=2)
    apply_style(ax, "A Potential PN Coefficients Error")
    plt.tight_layout()
    plt.savefig("visualizations/A_potential_terms.png")
    plt.close()

    # D Potential
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nus, e_D_0, label=r"$d_0$ (const)", lw=2)
    ax.plot(nus, e_D_1, label=r"$d_1$ ($u$)", color='gray', linestyle='--', alpha=0.7)
    ax.plot(nus, e_D_2, label=r"$d_2$ ($u^2$)", lw=2)
    ax.plot(nus, e_D_3, label=r"$d_3$ ($u^3$)", lw=2)
    apply_style(ax, "D Potential PN Coefficients Error")
    plt.tight_layout()
    plt.savefig("visualizations/D_potential_terms.png")
    plt.close()

    # Q Potential (Constant term only)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nus, e_Q_0, label=r"$z_3$", lw=2, color='tab:red')
    apply_style(ax, "Q Potential Coefficient Error")
    plt.tight_layout()
    plt.savefig("visualizations/Q_potential_terms.png")
    plt.close()

    # f Potential
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nus, e_f_0, label=r"$f_0$ (const)", lw=2)
    ax.plot(nus, e_f_1, label=r"$f_1$ ($x$)", lw=2)
    ax.plot(nus, e_f_2, label=r"$f_2$ ($x^2$)", lw=2)
    ax.plot(nus, e_f_3, label=r"$f_3$ ($x^3$)", lw=2)
    ax.plot(nus, e_f_3_l, label=r"$f_{3l}$ ($x^3 \ln x$)", lw=2, linestyle='--')
    apply_style(ax, "f Potential PN Coefficients Error")
    plt.tight_layout()
    plt.savefig("visualizations/f_potential_terms.png")
    plt.close()

    # 2. Combined Summary Plot (2x2 Grid)
    print(" - Rendering summary plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # A (Top Left)
    axes[0,0].plot(nus, e_A_0, label=r"$a_0$")
    axes[0,0].plot(nus, e_A_1, label=r"$a_1$")
    axes[0,0].plot(nus, e_A_3, label=r"$a_3$")
    axes[0,0].plot(nus, e_A_4, label=r"$a_4$")
    apply_style(axes[0,0], "A Potential")

    # D (Top Right)
    axes[0,1].plot(nus, e_D_0, label=r"$d_0$")
    axes[0,1].plot(nus, e_D_2, label=r"$d_2$")
    axes[0,1].plot(nus, e_D_3, label=r"$d_3$")
    apply_style(axes[0,1], "D Potential")

    # f (Bottom Left)
    axes[1,0].plot(nus, e_f_0, label=r"$f_0$")
    axes[1,0].plot(nus, e_f_1, label=r"$f_1$")
    axes[1,0].plot(nus, e_f_2, label=r"$f_2$")
    axes[1,0].plot(nus, e_f_3, label=r"$f_3$")
    axes[1,0].plot(nus, e_f_3_l, label=r"$f_{3l}$", linestyle='--')
    apply_style(axes[1,0], "f Potential")

    # Q (Bottom Right)
    axes[1,1].plot(nus, e_Q_0, label=r"$z_3$", color='tab:red')
    apply_style(axes[1,1], "Q Potential")

    plt.suptitle("Model Predicted PN Coefficients: Relative Error Across Mass Ratios", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("visualizations/PN_coefficients_summary.png")
    plt.close()
    
    print(f"\nSuccess! Plots saved to:\n - A_potential_terms.png\n - D_potential_terms.png\n - Q_potential_terms.png\n - f_potential_terms.png\n - PN_coefficients_summary.png")


main()
    
    

    
    
    
        
    
