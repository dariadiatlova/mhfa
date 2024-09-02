#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5
# Define the range of values for each parameter
batch_size=(128)
LLRD_factor=(0.8 1. 1.5)
LR_Transformer=(1e-3 2e-5)
LR_MHFA=(1e-3 5e-4 1e-4)
head_nb=(16 64 128)
test_gender=("M" "F")

# Iterate through all combinations of parameter values
for bs in "${batch_size[@]}"
do
  for llrd in "${LLRD_factor[@]}"
  do
    for lrt in "${LR_Transformer[@]}"
    do
      for mhfa in "${LR_MHFA[@]}"
      do
        for head in "${head_nb[@]}"
        do
            for tg in "${test_gender[@]}"
            do
              echo "Running script with batch_size=$bs, test_gender=$tg"

              # Run the Python script with the current parameter values
              python -m trainSERNet --config yaml/ser/Baseline.yaml --batch_size $bs --test_gender $tg \
              --LLRD_factor $llrd --LR_Transformer $lrt --LR_MHFA $mhfa --head_nb $head --distributed

              # Check if the script executed successfully
              if [ $? -ne 0 ]; then
                  echo "Script failed with batch_size=$bs, test_gender=$tg"
                  exit 1
              fi

              echo "Finished script with batch_size=$bs, test_gender=$tg"
            done
        done
      done
    done
  done
done

echo "All scripts executed successfully."