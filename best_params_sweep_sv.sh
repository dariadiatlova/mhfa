#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
# Define the range of values for each parameter
batch_size=32
LLRD_factor=1
LR_Transformer=8e-5
LR_MHFA=1e-3
weight_finetuning_reg=8e-4
seeds=(3 77 100 505 696)
save_ckpts=True
save_path="/app/data2/best_mhfa_sv_sweep_20/"

# Iterate through all combinations of parameter values
for seed in "${seeds[@]}"
do
    echo "Running script with eval seed=$seed"

    # Run the Python script with the current parameter values
    python -m trainSpeakerNet --config yaml/mhfa_vk.yaml --batch_size $batch_size --LR_MHFA $LR_MHFA \
    --LLRD_factor $LLRD_factor --LR_Transformer $LR_Transformer --weight_finetuning_reg $weight_finetuning_reg \
    --save_ckpts $save_ckpts --save_path $save_path --seed $seed

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Script failed with seed=$seed"
        exit 1
    fi

    echo "Finished script with seed=$seed"
done

echo "All scripts executed successfully."