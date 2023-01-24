#!/bin/bash


# run dataset
bash upload_script.sh dataset.py
kaggle kernels push
# run model
bash upload_script.sh model.py
kaggle kernels push

# run training
break=false

while [ "$break" = false ] 
do
    # wait until the other runs are completed
    dataset_run=$(kaggle kernels status davidfmora/dataset)
    model_run=$(kaggle kernels status davidfmora/model)

    if echo "$model_run" | grep -q "complete" && echo "$dataset_run" | grep -q "complete"; then
        bash upload_script.sh train.py
        kaggle kernels push
        rm kernel-metadata.json
        break=true
        echo "Experiment runned"

    elif echo "$model_run" | grep -q "error" || echo "$dataset_run" | grep -q "error"; then
        echo "Experiment failed"
        break=true
    fi

done
