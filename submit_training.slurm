#!/bin/bash
#SBATCH --job-name=train_eob_prelim
#SBATCH --partition=gpu_2day          # 'gpu_2day' is ideal for preliminary training up to 48 hours
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Request 1 task (for a single Python script)
#SBATCH --cpus-per-task=8             # Number of CPU cores for data loading/preprocessing
#SBATCH --gres=gpu:1                  # Request 1 GPU (you can increase this e.g., gpu:a100:1 if specific models are needed, but this cluster might just use gpu:1)
#SBATCH --mem=32G                     # System memory requested
#SBATCH --time=2-00:00:00             # Max time limit (2 days to match the partition limit)
#SBATCH --output=logs/train_%j.out    # Standard output log (creates a 'logs' dir)
#SBATCH --error=logs/train_%j.err     # Standard error log

# Make sure the logs directory exists for the slurm output
mkdir -p logs

echo "Starting training job on $HOSTNAME"
echo "Date: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# ==========================================================
# Environment Setup 
# ==========================================================
# Load any necessary HPC modules (uncomment and adjust versions as required)
# module load miniconda3
# module load cuda/12.1

# Activate your Python or Conda environment where JAX/PyTorch is installed
# source ~/.bashrc
# conda activate your_env_name

# ==========================================================
# Run Training
# ==========================================================
# Run the Python training script
# Note: Ensure you are submitting this script from /home/sidmahesh/EOB_NN_p4PN/
# Or you can explicitly cd into the directory:
# cd /home/sidmahesh/EOB_NN_p4PN

python -u EOB_NN_p4PN/EOB_NNp4PN/neural_eob_training_prelim.py

echo "Training job finished at $(date)"
