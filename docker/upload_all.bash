#!/usr/bin/env bash
# Define an array with exclusions in form ["conf", "wandb", "dataset", "__pycache__", "]
excludes=( "conf" "wandb" "dataset" "__pycache__" ".git" )

# Prefix each exclusion with "--exclude=" and sync excluding the specified directories/files
rsync -av "${excludes[@]/#/--exclude=}" --no-o --no-g ./ runpod:/workspace
# ./docker/upload.bash -j 32 -s dataset/recordings/ -d runpod:/workspace/dataset/recordings
# ./docker/upload.bash -j 8 -s conf/ -d runpod:/workspace/conf
