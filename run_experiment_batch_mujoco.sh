#!/bin/bash
set -e  # Exit if any command fails

# ===== Configuration =====
BUILD_CMD="cmake --build cmake-build-release --target ac_ppo_continuous_action -j 30"
EXECUTABLE="cmake-build-release/ac_ppo_continuous_action"
N=8   # Number of runs, adjust as needed
SEED_STEP=10000
# ==========================

echo "[INFO] Building project..."
eval $BUILD_CMD

#echo quit | nvidia-cuda-mps-control
echo "[INFO] Starting NVIDIA MPS..."
nvidia-cuda-mps-control -d


# Run N times with different seeds
for ((i=1; i<=N; i++)); do
    SEED=$((i * SEED_STEP))
    "$EXECUTABLE" --exp_name_stem Ant-v5_AC_PPO_Atari_Beta_Network_reward_manual_obs --seed $SEED &
done

wait
echo "[INFO] Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

echo "[INFO] All runs completed."