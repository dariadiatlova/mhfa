#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
# Define the range of values for each parameter
batch_size=(32)
LLRD_factor=(1. 1.2)
LR_Transformer=(8e-5 2e-5)
LR_MHFA=(1e-3 5e-4)
weight_finetuning_reg=(8e-4 1e-4)

# Iterate through all combinations of parameter values
for bs in "${batch_size[@]}"
do
  for llrd in "${LLRD_factor[@]}"
  do
    for lrt in "${LR_Transformer[@]}"
    do
      for mhfa in "${LR_MHFA[@]}"
      do
        for lambda in "${weight_finetuning_reg[@]}"
        do

          echo "Running script with batch_size=$bs"

          # Run the Python script with the current parameter values
          python -m trainSpeakerNet --config yaml/mhfa_vk.yaml --batch_size $bs \
          --LLRD_factor $llrd --LR_Transformer $lrt --LR_MHFA $mhfa --weight_finetuning_reg $lambda

          # Check if the script executed successfully
          if [ $? -ne 0 ]; then
              echo "Script failed with batch_size=$bs"
              exit 1
          fi

          echo "Finished script with batch_size=$bs"
        done
      done
    done
  done
done

echo "All scripts executed successfully."