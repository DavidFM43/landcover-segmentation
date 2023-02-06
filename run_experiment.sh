#!/bin/bash

# run dataset
bash upload_script.sh src/dataset.py
kaggle kernels push
# run model
bash upload_script.sh src/model.py
kaggle kernels push
# run utils
bash upload_script.sh src/utils.py
kaggle kernels push

break1=false

while [ "$break1" = false ] 
do
    # wait until the other runs are completed
    dataset_run=$(kaggle kernels status davidfmora/dataset)
    model_run=$(kaggle kernels status davidfmora/model)
    utils_run=$(kaggle kernels status davidfmora/utils)

    if echo "$model_run" | grep -q "complete" && echo "$dataset_run" | grep -q "complete" && echo "$utils_run" | grep -q "complete"; then
        echo "***Model and dataset run complete***"
        echo "***Running train***"
        bash upload_script.sh src/train.py
        kaggle kernels push
        rm kernel-metadata.json
        break1=true

    elif echo "$model_run" | grep -q "error"; then
        echo "***Model run failed***"
        echo ""
        kaggle kernels output davidfmora/model -q 
        grep "stderr" model.log | perl -ne 'print "$1\n" if /"data":"([^"]+)"/'
        rm model.log
        exit

    elif echo "$dataset_run" | grep -q "error"; then
        echo "***Dataset run failed***"
        echo ""
        kaggle kernels output davidfmora/dataset -q 
        grep "stderr" dataset.log | perl -ne 'print "$1\n" if /"data":"([^"]+)"/'
        rm dataset.log
        exit

    elif echo "$utils_run" | grep -q "error"; then
        echo "***Utils run failed***"
        echo ""
        kaggle kernels output davidfmora/utils -q 
        grep "stderr" utils.log | perl -ne 'print "$1\n" if /"data":"([^"]+)"/'
        rm utils.log
        exit

    fi

done

# run train and log
break2=false

while [ "$break2" = false ] 
do
    train_run=$(kaggle kernels status davidfmora/train)

    if echo "$train_run" | grep -q "complete"; then
        echo "***Experiment complete***"
        echo ""
        kaggle kernels output davidfmora/train -q
        grep "stdout" train.log | perl -ne 'print "$1\n" if /"data":"([^"]+)"/'
        echo ""
        perl -ne 'print "$1\n" if /"data":"([^"]+)"/' train.log | grep "Synced"
        break2=true

    elif echo "$train_run" | grep -q "error"; then
        echo "***Experiment failed***"
        echo ""
        kaggle kernels output davidfmora/train -q
        grep "stderr" train.log | perl -ne 'print "$1\n" if /"data":"([^"]+)"/'
        break2=true
    fi
     
done
