#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
# Define the range of values for each parameter
accumulate_grad_each_n_step=(8 16 32)
LLRD_factor=(0.8 1. 1.2)
LR_Transformer=(1e-4 2e-5)
LR_MHFA=(1e-3 5e-4)
head_nb=(16 64)
test_gender=("M" "F")

# Iterate through all combinations of parameter values
for acc in "${accumulate_grad_each_n_step[@]}"
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
              echo "Running script with accumulate_grad_each_n_step=$acc, test_gender=$tg"

              # Run the Python script with the current parameter values
              python -m trainSERNet --config yaml/ser/Baseline.yaml --accumulate_grad_each_n_step $acc \
              --test_gender $tg --LLRD_factor $llrd --LR_Transformer $lrt --LR_MHFA $mhfa --head_nb $head

              # Check if the script executed successfully
              if [ $? -ne 0 ]; then
                  echo "Script failed with accumulate_grad_each_n_step=$acc, test_gender=$tg"
                  exit 1
              fi

              echo "Finished script with accumulate_grad_each_n_step=$acc, test_gender=$tg"
            done
        done
      done
    done
  done
done

echo "All scripts executed successfully."