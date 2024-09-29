#!/bin/bash
#SBATCH --job-name=image_run_container_mount
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=image_run_%j.out
#SBATCH --error=image_run_%j.err
#SBATCH --nodelist=ml5-3

DEVICE=$1
IMAGE="/fast_nfs/images/diatlova+mhfa-slurm.sqsh"
CODE_DIR="/fast_nfs/d.dyatlova/repos/mhfa"
DATA_DIR="/fast_nfs/voice_datasets/VoxCeleb1"
CONTAINER_CODE_DIR="/app"
CONTAINER_DATA_DIR="/app/data"
XDG_DATA_HOME="/fast_nfs/d.dyatlova/cache"

export XDG_DATA_HOME=$XDG_DATA_HOME


# Set execute permissions on the script
chmod +x $CODE_DIR/grid_search_sv_64.sh

srun --exclusive --gres=gpu:1 --container-image="$IMAGE" \
     --container-mounts="$CODE_DIR:$CONTAINER_CODE_DIR,$DATA_DIR:$CONTAINER_DATA_DIR" \
     --container-env="XDG_DATA_HOME=$XDG_DATA_HOME" \
     /bin/bash -c "export CUDA_VISIBLE_DEVICES=$DEVICE && cd $CONTAINER_CODE_DIR && ./grid_search_sv_64.sh"

## Function to run the grid_search_sv_64.sh script on a specific GPU
#run_grid_search() {
#    local gpu_ids=$1
#    srun --exclusive --gres=gpu:2 --container-image="$IMAGE" \
#         --container-mounts="$CODE_DIR:$CONTAINER_CODE_DIR,$DATA_DIR:$CONTAINER_DATA_DIR" \
#         --container-env="XDG_DATA_HOME=$XDG_DATA_HOME" \
#         /bin/bash -c "export CUDA_VISIBLE_DEVICES=$gpu_ids && cd $CONTAINER_CODE_DIR && ./grid_search_sv_64.sh"
#}
#
## Launch 8 instances of the grid_search_sv_64.sh script, each using a different GPU
#for gpu_id in {0..3}; do
#    gpu_ids=$((2 * i)),$((2 * i + 1))
#    run_grid_search $gpu_ids &
#done

wait