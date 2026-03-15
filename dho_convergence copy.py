import numpy as np
import matplotlib.pyplot as plt

def tanh(x): return np.tanh(x)
def dtanh(x): return 1.0 - np.tanh(x)**2

# Hyperparameters
H = 4
lr = 0.02
epochs = 5000
num_samples = 800
beta_f = 0.4
beta_low, beta_high = 0.2, 0.6

# 1. Data Generation
q = np.random.uniform(-1, 1, (num_samples, 1))
p = np.random.uniform(-1, 1, (num_samples, 1))
b = np.random.uniform(beta_low, beta_high, (num_samples, 1))

# Targets for potential gradients
target_v_grad = q
target_d0_grad = beta_f * p
target_d1_grad = b * p

# 2. Weights Initialization (Xavier-like for tanh)
def init_net(in_dim):
    return {
        'W1': np.random.randn(in_dim, H) * np.sqrt(1/in_dim),
        'W2': np.random.randn(H, 1) * np.sqrt(1/H),
        'b1': np.zeros((1, H))
    }

m0_v, m0_d = init_net(1), init_net(1)
m1_v, m1_d = init_net(1), init_net(2) # D1 takes (p, beta)

# History trackers for V and D components specifically
v0_err, d0_err = [], []
v1_err, d1_err = [], []

for i in range(epochs):
    # --- MODEL 0: Fixed Beta ---
    z_v0 = q @ m0_v['W1'] + m0_v['b1']
    grad_v0 = dtanh(z_v0) * (m0_v['W2'].T * m0_v['W1'].T) 
    grad_v0_sum = np.sum(grad_v0, axis=1, keepdims=True)
    
    z_d0 = p @ m0_d['W1'] + m0_d['b1']
    grad_d0 = dtanh(z_d0) * (m0_d['W2'].T * m0_d['W1'].T)
    grad_d0_sum = np.sum(grad_d0, axis=1, keepdims=True)
    
    res0 = (-grad_v0_sum - grad_d0_sum) - (-q - beta_f * p)
    
    # Track Convergence of Components
    v0_err.append(np.mean((grad_v0_sum - target_v_grad)**2))
    d0_err.append(np.mean((grad_d0_sum - target_d0_grad)**2))
    
    # Simple SGD update for Model 0
    common0 = 2 * res0 / num_samples
    # Update V weights
    m0_v['W2'] -= lr * (dtanh(z_v0).T @ (-common0 * m0_v['W1'].T)).T.sum(axis=1, keepdims=True).T
    m0_v['W1'] -= lr * (common0.T @ (-dtanh(z_v0) * m0_v['W2'].T))
    # Update D0 weights
    m0_d['W2'] -= lr * (dtanh(z_d0).T @ (-common0 * m0_d['W1'].T)).T.sum(axis=1, keepdims=True).T
    m0_d['W1'] -= lr * (common0.T @ (-dtanh(z_d0) * m0_d['W2'].T))

    # --- MODEL 1: Variable Beta ---
    z_v1 = q @ m1_v['W1'] + m1_v['b1']
    grad_v1 = dtanh(z_v1) * (m1_v['W2'].T * m1_v['W1'].T)
    grad_v1_sum = np.sum(grad_v1, axis=1, keepdims=True)
    
    inputs_d1 = np.hstack([p, b])
    z_d1 = inputs_d1 @ m1_d['W1'] + m1_d['b1']
    # Partial derivative wrt p uses only the first row of W1
    grad_d1 = dtanh(z_d1) * (m1_d['W2'].T * m1_d['W1'][0:1, :])
    grad_d1_sum = np.sum(grad_d1, axis=1, keepdims=True)
    
    res1 = (-grad_v1_sum - grad_d1_sum) - (-q - b * p)
    
    v1_err.append(np.mean((grad_v1_sum - target_v_grad)**2))
    d1_err.append(np.mean((grad_d1_sum - target_d1_grad)**2))
    
    common1 = 2 * res1 / num_samples
    m1_v['W2'] -= lr * (dtanh(z_v1).T @ (-common1 * m1_v['W1'].T)).T.sum(axis=1, keepdims=True).T
    m1_v['W1'] -= lr * (common1.T @ (-dtanh(z_v1) * m1_v['W2'].T))
    
    m1_d['W2'] -= lr * (dtanh(z_d1).T @ (-common1 * m1_d['W1'][0:1, :])).T.sum(axis=1, keepdims=True).T
    m1_d['W1'] -= lr * (inputs_d1.T @ (common1 * -dtanh(z_d1) @ m1_d['W2'].T))

# Plotting Results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(v0_err, label='$V(q)$ Potential')
plt.plot(d0_err, label='$D_0(p)$ Potential', linestyle='--')
plt.yscale('log'); plt.title('$L_0$ (Fixed $\\beta$) Convergence'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(v1_err, label='$V(q)$ Potential')
plt.plot(d1_err, label='$D_1(p, \\beta)$ Potential', linestyle='--')
plt.yscale('log'); plt.title('$L_1$ (Variable $\\beta$) Convergence'); plt.legend(); plt.grid(True)
plt.show()