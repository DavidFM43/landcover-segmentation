#!/bin/bash

script_name=$1
file_name=$(basename "$script_name")
file_stem=${file_name%.*}

if [ "$file_stem" == "train" ]; then
json='{
  "id": "davidfmora/'$file_stem'",
  "title": "'$file_stem'",
  "code_file": "'$script_name'",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "balraj98/deepglobe-land-cover-classification-dataset", 
    "davidfmora/processed-masks"
  ],
  "competition_sources": [],
  "kernel_sources": [
    "davidfmora/dataset",
    "davidfmora/model",
    "davidfmora/utils"
  ]
}'
else 
json='{
  "id": "davidfmora/'$file_stem'",
  "title": "'$file_stem'",
  "code_file": "'$script_name'",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "keywords": [
    "util-script"
  ],
  "dataset_sources": [
    "balraj98/deepglobe-land-cover-classification-dataset", 
    "davidfmora/processed-masks"
  ],
  "competition_sources": [],
  "kernel_sources": []
}'
fi
echo $json > kernel-metadata.json

