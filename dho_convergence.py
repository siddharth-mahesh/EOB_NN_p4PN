import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
k_max = 3
l_max = 1
lr = 0.005 # Small for stability
num_epochs = 5000
num_samples = 1000

# Physics constants
beta_val = 0.4
beta_range = (0.1, 0.7)

# Data Generation
q = np.random.uniform(-1, 1, num_samples)
p = np.random.uniform(-1, 1, num_samples)
b_vec = np.random.uniform(beta_range[0], beta_range[1], num_samples)

# Weights Initialization
c0, s0 = np.zeros(k_max + 1), np.zeros(k_max + 1)
c1 = np.zeros(k_max + 1)
g1 = np.zeros((k_max + 1, l_max + 1))

h0, h1 = [], []
w1_c2, w1_g21 = [], []

for epoch in range(num_epochs):
    # --- Model 0 (Fixed Beta) ---
    # Prediction: p_dot = -(2*c2*q + 3*c3*q^2) - (2*s2*p + 3*s3*p^2)
    # Target: p_dot = -q - beta*p
    p_dot_pred0 = -(2*c0[2]*q + 3*c0[3]*q**2) - (2*s0[2]*p + 3*s0[3]*p**2)
    res0 = p_dot_pred0 - (-q - beta_val * p) # (Pred - True)
    
    # Loss = <res^2>
    # Grad = 2 * <res * d_pred/dw>
    c0[2] -= lr * np.mean(res0 * (-2*q))
    c0[3] -= lr * np.mean(res0 * (-3*q**2))
    s0[2] -= lr * np.mean(res0 * (-2*p))
    s0[3] -= lr * np.mean(res0 * (-3*p**2))
    h0.append(np.mean(res0**2))

    # --- Model 1 (Variable Beta) ---
    p_dot_pred1 = -(2*c1[2]*q + 3*c1[3]*q**2) - (2*g1[2,1]*p*b_vec)
    res1 = p_dot_pred1 - (-q - b_vec * p)
    
    c1[2] -= lr * np.mean(res1 * (-2*q))
    c1[3] -= lr * np.mean(res1 * (-3*q**2))
    g1[2,1] -= lr * np.mean(res1 * (-2*p*b_vec))
    
    h1.append(np.mean(res1**2))
    w1_c2.append(c1[2])
    w1_g21.append(g1[2,1])

# Plotting results
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(h0, label='$L_0$ (Fixed)')
ax[0].plot(h1, label='$L_1$ (Variable)')
ax[0].set_yscale('log')
ax[0].set_title('Loss Convergence (Corrected)')
ax[0].set_xlabel('Epoch'); ax[0].legend(); ax[0].grid(True)

ax[1].axhline(0.5, color='red', linestyle='--', label='Target (0.5)')
ax[1].plot(w1_c2, label='$c_2$ (Potential)')
ax[1].plot(w1_g21, label='$\\gamma_{21}$ (Dissipation)')
ax[1].set_title('Weight Convergence ($L_1$)')
ax[1].set_xlabel('Epoch'); ax[1].legend(); ax[1].grid(True)

plt.tight_layout()
plt.savefig('final_convergence.png')

print(f"Final Weights L1: c2={c1[2]:.4f}, g21={g1[2,1]:.4f}")
print(f"Final Loss L1: {h1[-1]:.6e}")