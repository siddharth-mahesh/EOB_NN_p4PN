# schwarzschild_hamiltonian.py
"""
    Code to calibrate the pseudo-4PN EOB Hamiltonians with gravitational wave data.
    The module generates training waveforms from the SXS waveform database.
    The module then trains a Hamiltonian Neural Network (HNN) to learn the pseudo-4PN component of the EOB Hamiltonian.
"""
# Core libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Data generation class
import hnn_data_generation as train_class

# HNN Model class
import hnn_class as hnn_class

# --- Training and Evaluation Function ---
def train_and_evaluate_model(model, hyperparams, train_ds, val_ds):
    """
    Compile and trains the HNN model.

    Args:
        model: The HNN model to train.
        hyperparams: Dictionary of hyperparameters for training. Elements are
            hidden_dim: Dimension of the hidden layers.
            num_layers: Number of hidden layers.
            learning_rate: Learning rate for the optimizer.
        train_ds: Training dataset.
        val_ds: Validation dataset.
    """
    print(f"\n--- Training with Hyperparameters: {hyperparams} ---")

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    
    # The loss function is the mean squared relative error
    # This is a good choice for relative error as it is scale invariant
    def msre(y_true, y_pred):
        return tf.reduce_mean(tf.square((y_true - y_pred)/y_true))
    
    # Compile the model
    model.compile(optimizer=optimizer,loss=msre)

    # Early stopping callback that stops training when validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-9,
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    num_epochs = 50
    x_train , y_train = train_ds
    x_val , y_val = val_ds
    history = model.fit(
        x_train, y_train,
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )
    final_val_loss = history.history['val_loss'][-1]
    print(f"Result -> Final Validation Loss: {final_val_loss:.6f}")

    return final_val_loss, model, history

# --- Visualization ---
def plot_potential(model, grid_size=100, r_range=(7, 200),train_range=(7, 35)):
    """
    Visualizes the learned Hamiltonian potential A(r) vs. the true potential.

    Args:
        model: The HNN model.
        grid_size: Number of points to use for the radial grid.
        r_range: Range of r values to use for the grid (min, max).
        train_range: Range of r values used for training (min, max).
    """
    # Create a grid of r values
    r_vals = tf.constant(np.linspace(r_range[0], r_range[1], grid_size), dtype=tf.float32)
    r_in = tf.expand_dims(r_vals, axis=-1)
    
    # Compute the learned potential and true potential
    learned_potentials = model.hamiltonian_net(r_in).numpy().squeeze()
    true_potentials = 1 - 2 / r_vals.numpy()

    # Plot the learned potential and true potential
    plt.figure(figsize=(8, 6))
    plt.plot(r_vals, learned_potentials, label='Learned A(r)', lw=2)
    plt.plot(r_vals, true_potentials, label='True A(r) = 1 - 2/r', linestyle='--', color='red')
    plt.axvline(x=train_range[0], color='green', linestyle=':', label='Training Range Start')
    plt.axvline(x=train_range[1], color='green', linestyle=':', label='Training Range End')

    plt.xlabel('r (Radius)')
    plt.ylabel('Potential A(r)')
    plt.title('Learned Hamiltonian Potential vs. True Potential')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learned_potential_schwarzschild.png')
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    noise_level = 0.0
    mass_ratio = 1e-5
    sym_mass_ratio = mass_ratio / (1 + mass_ratio)**2
    l_isco = 12
    num_radii = 50
    num_points = 128
    
    # Stavle circular orbit initial conditions
    r_vals = np.linspace(7, 35, num_radii)
    ls = np.sqrt(r_vals**2/(r_vals - 3))
    t_start = 0.0
    t_p = 2 * np.pi * (r_vals[-1]**(1.5))
    t_end = 1 * t_p

    initial_conditions = tf.constant([
        [r_vals[i], 0.0, 0.0, ls[i]] for i in range(num_radii)
    ], dtype=tf.float32)

    omega_ref = 2

    # 1. Generate Training Data
    print("1. Generating training data...")
    generator = train_class.HamiltonianDataGenerator(nu=sym_mass_ratio, omega_ref=omega_ref)
    time_points, trajectory, h22_complex = generator.generate_data(
        t_span=[t_start, t_end],
        t_points=num_points,
        x0=initial_conditions,
        noise_std=noise_level
    )

    # 2. Prepare Data by reshaping and splitting into training and validation sets
    print("2. Preparing data for model training...")
    z_data = tf.reshape(trajectory, (-1, 4))
    h_data_reshaped = tf.reshape(tf.abs(h22_complex), (-1, 1))
    val_size = len(z_data) // 5
    z_train = z_data[:-val_size]
    h_train = h_data_reshaped[:-val_size]
    z_val = z_data[-val_size:]
    h_val = h_data_reshaped[-val_size:]

    # 3. Build and Train the Model
    print("3. Building and training the HNN model...")
    hyperparams = {'hidden_dim': 50, 'num_layers': 2, 'learning_rate': 1e-3}
    # hyperparameters for the HNN model
    model_hyperparams = {'hidden_dim': 50, 'num_layers': 2}

    hnn_model = hnn_class.HNN_Model(nu=sym_mass_ratio, **model_hyperparams)
    
    # Train the model
    final_val_loss, trained_model, history = train_and_evaluate_model(
        model=hnn_model,
        hyperparams=hyperparams,
        train_ds=(z_train, h_train),
        val_ds=(z_val, h_val)
    )

    # 4. Visualize Results
    print("4. Visualizing learned physical potential...")
    plot_potential(trained_model, r_range=(3, 50),train_range=(7, 35))

    # 5. Save the model
    print("5. Saving the model...")
    trained_model.save('schwarzschild_hnn_model.keras')
    
