#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
# Define the range of values for each parameter
accumulate_grad_each_n_step=32
LLRD_factor=1.2
LR_Transformer=2e-5
LR_MHFA=1e-3
head_nb=64
eval_session=("Ses01" "Ses02" "Ses03" "Ses04" "Ses05")
test_gender=("M" "F")
save_ckpts=True
save_path="/app/nfs_small/full_iemocap_best_mhfa/"

# Iterate through all combinations of parameter values
for session in "${eval_session[@]}"
do
    for tg in "${test_gender[@]}"
    do
      echo "Running script with eval session=$session, test_gender=$tg"

      # Run the Python script with the current parameter values
      python -m trainSERNet --config yaml/ser/Baseline.yaml --accumulate_grad_each_n_step $accumulate_grad_each_n_step \
      --LLRD_factor $LLRD_factor --LR_Transformer $LR_Transformer --LR_MHFA $LR_MHFA --head_nb $head_nb --test_gender $tg \
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