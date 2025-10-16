import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from typing import Dict, Tuple, List

from dho_dhnn import DHNN_Model
from dho_data import DampedHarmonicOscillator

def train_dhnn_model(train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], model_params: Dict[str, int], training_params: Dict[str, int]):
    """Trains the DHNN model using the provided data and parameters.

    Args:
        train_data (Tuple[np.ndarray, np.ndarray]): The training data.
        val_data (Tuple[np.ndarray, np.ndarray]): The validation data.
        model_params (Dict[str, int]): The model parameters.
        training_params (Dict[str, int]): The training parameters.

    Returns:
        DHNN_Model: The trained model.
    """
    model = DHNN_Model(**model_params)
    x_train , y_train = train_data
    x_val , y_val = val_data
    optimizer = Adam(learning_rate=training_params['learning_rate'])
    # use mean squared error separately on coordinate and momentum derivatives
    def custom_mse(y_true, y_pred):
        qdot_true , pdot_true = tf.split(y_true, 2, axis=-1)
        qdot_pred , pdot_pred = tf.split(y_pred, 2, axis=-1)
        mse_qdot = tf.reduce_mean(tf.square((qdot_true - qdot_pred)))
        mse_pdot = tf.reduce_mean(tf.square((pdot_true - pdot_pred)))
        loss = mse_qdot + mse_pdot
        return loss
    model.compile(optimizer=optimizer, loss=custom_mse)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=training_params['epochs'], batch_size=training_params['batch_size'], validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])
    return model

def plot_results(t_eval, true_y, predicted_y):
    """Plots the trajectory and phase space comparisons.

    Args:
        t_eval (np.ndarray): The time points.
        true_y (np.ndarray): The true trajectory.
        predicted_y (np.ndarray): The predicted trajectory.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Trajectory Plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.plot(t_eval, true_y[:, 0], 'b-', label='True Position (q)', linewidth=2)
    ax1.plot(t_eval, true_y[:, 1], 'r-', label='True Momentum (p)', linewidth=2)
    ax1.plot(t_eval, predicted_y[:, 0], 'c--', label='D-HNN Predicted Position (q)', linewidth=2)
    ax1.plot(t_eval, predicted_y[:, 1], 'm--', label='D-HNN Predicted Momentum (p)', linewidth=2)
    ax1.set_title('Trajectory Prediction: True vs. D-HNN', fontsize=16)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    fig1.tight_layout()
    plt.savefig('trajectory_prediction_dho.png', dpi=300)
    plt.show()

    # Phase Space Plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.plot(true_y[:, 0], true_y[:, 1], 'b-', label='True Trajectory', linewidth=2)
    ax2.plot(predicted_y[:, 0], predicted_y[:, 1], 'r--', label='D-HNN Predicted Trajectory', linewidth=2)
    ax2.set_title('Phase Space Portrait', fontsize=16)
    ax2.set_xlabel('Position (q)', fontsize=12)
    ax2.set_ylabel('Momentum (p)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    ax2.set_aspect('equal', 'box')
    fig2.tight_layout()
    plt.savefig('phase_space_portrait_dho.png', dpi=300)
    plt.show()

def plot_potentials(model: DHNN_Model, data: DampedHarmonicOscillator, q_range=(-2.5, 2.5), p_range=(-2.5, 2.5), grid_size=50):
    """Visualizes the learned and true potentials.

    Args:
        model (DHNN_Model): The trained DHNN model.
        data (DampedHarmonicOscillator): The Damped Harmonic Oscillator data.
        q_range (tuple, optional): The range of q values. Defaults to (-2.5, 2.5).
        p_range (tuple, optional): The range of p values. Defaults to (-2.5, 2.5).
        grid_size (int, optional): The number of grid points. Defaults to 50.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    q = np.linspace(q_range[0], q_range[1], grid_size)
    p = np.linspace(p_range[0], p_range[1], grid_size)
    Q, P = np.meshgrid(q, p)
    grid_points = np.stack([Q.flatten(), P.flatten()], axis=-1)
    _ , P_input = tf.split(grid_points, 2, axis=-1)
    H = model.hnn(tf.constant(grid_points, dtype=tf.float32)).numpy()
    D = model.dnn(tf.constant(P_input, dtype=tf.float32)).numpy()
    H = H.reshape(grid_size, grid_size)
    D = D.reshape(grid_size, grid_size)

    # True Potentials
    H_true , D_true = data._hamiltonian_and_dissipative_potential(tf.constant(grid_points, dtype=tf.float32))
    H_true = H_true.numpy().reshape(grid_size, grid_size)
    D_true = D_true.numpy().reshape(grid_size, grid_size)
    

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle('Learned vs. True Potentials', fontsize=20)
    
    # Learned Hamiltonian
    c1 = axes[0, 0].contourf(Q, P, H, levels=20, cmap='viridis')
    fig.colorbar(c1, ax=axes[0, 0])
    axes[0, 0].set_title('Learned Hamiltonian (H)', fontsize=14)
    axes[0, 0].set_xlabel('q')
    axes[0, 0].set_ylabel('p')
    
    # Learned Dissipation Potential
    c2 = axes[0, 1].contourf(Q, P, D, levels=20, cmap='inferno')
    fig.colorbar(c2, ax=axes[0, 1])
    axes[0, 1].set_title('Learned Dissipation Potential (D)', fontsize=14)
    axes[0, 1].set_xlabel('q')
    axes[0, 1].set_ylabel('p')

    # True Hamiltonian
    c3 = axes[1, 0].contourf(Q, P, H_true, levels=20, cmap='viridis')
    fig.colorbar(c3, ax=axes[1, 0])
    axes[1, 0].set_title('True Hamiltonian (H)', fontsize=14)
    axes[1, 0].set_xlabel('q')
    axes[1, 0].set_ylabel('p')

    # True Dissipation Potential
    c4 = axes[1, 1].contourf(Q, P, D_true, levels=20, cmap='inferno')
    fig.colorbar(c4, ax=axes[1, 1])
    axes[1, 1].set_title(f'True Dissipation Potential (D, for b={data.b})', fontsize=14)
    axes[1, 1].set_xlabel('q')
    axes[1, 1].set_ylabel('p')

    plt.savefig('learned_potentials_dho.png', dpi=300)
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    train_flag = True
    if train_flag:
        # --- Configuration ---
        DAMPING_COEFFICIENT = 0.1
        NOISE = 0.0
        DAMPING_TIME_SCALE = 1/DAMPING_COEFFICIENT
        TIME_SPAN = [0, 10*DAMPING_TIME_SCALE]
        TIME_POINTS = 1000
        rng = np.random.default_rng()
        num_grid_pts = 4000
        qs = rng.uniform(-2, 2, num_grid_pts)
        ps = rng.uniform(-2, 2, num_grid_pts)
        INITIAL_CONDITIONS = tf.constant([
            [qs[i], ps[i]] for i in range(num_grid_pts)
        ], dtype=tf.float32)
        LEARNING_RATE = 1e-2
        EPOCHS = 5000
        BATCH_SIZE = num_grid_pts//128
        model_params = {'hidden_dim': 256, 'num_layers': 1}
        # --- Data Generation ---
        print("1. Generating and preparing data...")
        data_class = DampedHarmonicOscillator(DAMPING_COEFFICIENT)
        x_data = INITIAL_CONDITIONS
        y_data = data_class._dynamics(INITIAL_CONDITIONS)

        print(f"x_data shape: {x_data.shape} expected ({num_grid_pts},2)")
        print(f"y_data shape: {y_data.shape} expected ({num_grid_pts},2)")
        
        dt = TIME_SPAN[1] - TIME_SPAN[0]
        
        val_grid_points = num_grid_pts//5
        qv = rng.uniform(-2, 2, val_grid_points)
        pv = rng.uniform(-2, 2, val_grid_points)
        INITIAL_CONDITIONS_VAL = tf.constant([
            [qv[i], pv[i]] for i in range(val_grid_points)
        ], dtype=tf.float32)
        val_x_data = INITIAL_CONDITIONS_VAL
        val_y_data = data_class._dynamics(INITIAL_CONDITIONS_VAL)
        
        # --- Model Training ---
        print("2. Training the model...")
        d_hnn_model = train_dhnn_model((x_data, y_data), (val_x_data, val_y_data), model_params, {'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE})
        print("3. Saving the model...")
        d_hnn_model.save('d_hnn_model_dho.keras')
    else:
        d_hnn_model = tf.keras.models.load_model('d_hnn_model_dho.keras')
    # --- Plotting Results ---
    print("4. Plotting results...")
    qtest = rng.uniform(-2, 2, 2)
    ptest = rng.uniform(-2, 2, 2)
    INITIAL_CONDITIONS_TEST = tf.constant([
        [qtest[i], ptest[i]] for i in range(2)
    ], dtype=tf.float32)
    
    t_eval = np.linspace(TIME_SPAN[0], TIME_SPAN[1], TIME_POINTS)
    true_traj , _ = data_class._leapfrog_integrator(INITIAL_CONDITIONS_TEST, TIME_SPAN, TIME_POINTS)
    predicted_traj , _ = d_hnn_model._leapfrog_integrator(INITIAL_CONDITIONS_TEST, TIME_SPAN, TIME_POINTS)

    # pick the second trajectory to plot. traj has shape (TIME_POINTS, 2, 2)
    true_traj = true_traj[:,1,:]
    predicted_traj = predicted_traj[:,1,:]
    
    plot_results(t_eval, true_traj, predicted_traj)
    
    print("5. Visualizing learned physical potentials...")
    plot_potentials(model=d_hnn_model, data=data_class)

    print("\nTraining complete.")