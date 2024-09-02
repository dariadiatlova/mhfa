#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5
# Define the range of values for each parameter
batch_size=128
LLRD_factor=1.
LR_Transformer=2e-5
LR_MHFA=5e-4
head_nb=16
eval_session=("Ses01" "Ses02" "Ses03" "Ses04" "Ses05")
test_gender=("M" "F")
save_ckpts=True
save_path="/app/data2/best_mhfa_sweep/"

# Iterate through all combinations of parameter values
for session in "${eval_session[@]}"
do
    for tg in "${test_gender[@]}"
    do
      echo "Running script with eval session=$session, test_gender=$tg"

      # Run the Python script with the current parameter values
      python -m trainSERNet --config yaml/ser/Baseline.yaml --batch_size $batch_size --test_gender $tg \
      --LLRD_factor $LLRD_factor --LR_Transformer $LR_Transformer --LR_MHFA $LR_MHFA --head_nb $head_nb \
      --eval_session $session --save_ckpts $save_ckpts --save_path $save_path --distributed

      # Check if the script executed successfully
      if [ $? -ne 0 ]; then
          echo "Script failed with batch_size=$bs, test_gender=$tg"
          exit 1
      fi

      echo "Finished script with eval session=$session, test_gender=$tg"
    done
done

echo "All scripts executed successfully."