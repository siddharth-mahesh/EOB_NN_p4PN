# EOB_NN_p4PN

A working repository for Effective-One-Body (EOB) dynamics and waveform modeling augmented with neural/physics-informed Dissipative Hamiltonian neural networks (HNNs). It includes analytic 3PN EOB components, code-generation utilities, example training scripts, and pre-generated data/models.

## Installation

- **Prerequisites**
  - Python 3.10 or 3.11
  - Recommended: a virtual environment (venv or conda)

- **Clone and set up environment**

```bash
git clone https://github.com/sidmahesh/EOB_NN_p4PN.git
cd EOB_NN_p4PN

# Create and activate a virtual environment (venv example)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

- **Optional: JAX with GPU**
  - The repository uses JAX. For GPU acceleration, install the matching CUDA/cuDNN build of `jax`/`jaxlib` from the official instructions: https://jax.readthedocs.io/en/latest/installation.html
  - If you install GPU wheels, do it before (or instead of) the CPU wheels pinned in `requirements.txt`.

## Repository structure

- (In Progress)**EOB_NNp4PN/**
  - Core EOB + NN code and utilities.
  - Notable files:
    - `EOB_NNp4PN.py` Main entry for p4PN EOB with NN components.
    - `eob_p4pn.py`, `eob3pn.py`, `eob3pn_tf.py` EOB dynamics (p4PN/3PN, incl. a TensorFlow variant).
    - `gamma.py` Auxiliary functions/constants for EOB.
    - `hnn_class.py` HNN model definition utilities.
    - `hnn_data_generation.py` Data generation for training HNNs.
    - `schwarzschild_hnn_model.keras` Example pre-trained Keras model.

- **EOB_3PN/**
  - Baseline/analytic 3PN EOB reference (`eob3pn.py`, `gamma.py`).

- **codegen_3pn/**
  - Symbolic/code-generation utilities and generated 3-3.5PN accurate expressions.
- **dho_example/**
  - Damped Harmonic Oscillator examples demonstrating DHNN training and hyperparameter optimization.
  - Includes small data files (`x_*.dat`, `y_*.dat`) and scripts using Equinox/Optuna.

- **diffrax_example.py**
  - Minimal example of JAX-based ODE integration with Diffrax.

- **requirements.txt**, **LICENSE**, **README.md**
  - Project metadata and dependency pins.

## Quick start

- **Run a toy DHNN training example (Dissipative Harmonic Oscillator)**

```bash
python dho_example/damped_harmonic_oscillator_train_equinox.py
# this trains thhe DHNN for a choice of hyperparameters
# or
python dho_example/damped_harmonic_oscillator_train_optuna.py # this does hyperparameter optimization with Optuna
```

- **Generate JAX-differentiable 3PN-accurate EOB waveforms**

```bash
python EOB_3PN/eob3pn.py
```

## Notes

- This repository is an active work-in-progress; APIs and scripts may change.
- If you encounter environment issues, verify your Python version and JAX installation, especially for GPU setups.

## License

This project is licensed under the terms of the BSD 2-Clause License. See the LICENSE file included in the repository.
