#!/bin/bash
#SBATCH --job-name=train_icicle
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu,pi_ccoley
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=magled@mit.edu

cd /home/magled/icicle-dev
source .env # WANDB_API_KEY=... -- workaround since currently, only 86 char long API keys are provided.
module load miniforge cuda/12.4 gcc/12.2.0
export WANDB_PROJECT="icicle-dev"
export WANDB_ENTITY="mlederbauer"
export TORCH_CPP_LOG_LEVEL="ERROR"

export GCC_LIB_DIR=/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64
export LD_LIBRARY_PATH=$GCC_LIB_DIR:$LD_LIBRARY_PATH

# Ensure CUDA_VISIBLE_DEVICES is set from SLURM GPU allocation
# (some nodes/configs don't set this automatically with --gres)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    for var in SLURM_STEP_GPUS SLURM_JOB_GPUS GPU_DEVICE_ORDINAL; do
        val=$(eval echo \$$var)
        if [ -n "$val" ]; then
            export CUDA_VISIBLE_DEVICES=$val
            break
        fi
    done
fi

# Debug GPU allocation
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi -L

# Signal handler for preemptable node termination
# This allows graceful shutdown and checkpoint saving before job is killed
handle_preemption() {
    echo "Received preemption signal (SIGUSR1) at $(date)"
    echo "Job will be terminated soon, saving checkpoint..."
    # Send SIGTERM to the running process to trigger checkpoint saving
    if [ ! -z "$CHILD_PID" ]; then
        kill -TERM $CHILD_PID
        wait $CHILD_PID
    fi
    exit 0
}

# Register signal handler
trap handle_preemption SIGUSR1

# Run whatever command was passed as arguments
eval "$@"