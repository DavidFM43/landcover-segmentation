#!/bin/bash

script_name=$1
file_stem=${script_name%.*}

cat > kernel-metadata.json << EOL
{
  "id": "davidfmora/$file_stem",
  "title": "$file_stem",
  "code_file": "$script_name",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOL

kaggle kernels push

