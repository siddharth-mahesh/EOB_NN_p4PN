#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=optuna_hpo     # Job name
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sm0193@mix.wvu.edu
#SBATCH --ntasks=1                # Each worker is a single task
#SBATCH --array=1-20
#SBATCH --output=/users/sm0193/scratch/EOB_NN_p4PN/dho_example/logs/worker_%A_%a.out
#SBATCH --error=/users/sm0193/scratch/EOB_NN_p4PN/dho_example/logs/worker_%A_%a.err

# output and error logs
PROJECT_DIR="/users/sm0193/scratch/EOB_NN_p4PN/dho_example"
LOG_DIR="${PROJECT_DIR}/logs"

# ---
# Environment Setup
# ---
echo "Starting Optuna worker $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Project directory: $PROJECT_DIR"
cd $PROJECT_DIR
module load lang/python/cpython_3.11.3_gcc122
source /users/sm0193/scratch/EOB_NN_p4PN/.venv/bin/activate

# ---
# HPO script
# ---
echo "Running optimization script..."
python damped_harmonic_oscillator_train_optuna_cluster.py

echo "Worker $SLURM_ARRAY_TASK_ID finished."
