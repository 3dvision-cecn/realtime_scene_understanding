#! /usr/bin/bash

# Get CLUSTER_USER from command line argument
CLUSTER_USER=${2:-$USER}

# Set environment variables for proper config directory handling
export MPLCONFIGDIR=/tmp/matplotlib
export YOLO_CONFIG_DIR=/tmp/ultralytics
export DISPLAY=${DISPLAY:-:0}
export HF_HOME=/cluster/work/cvg/students/nec/.cache/huggingface

singularity exec \
 -B /cluster/work/cvg/students/$CLUSTER_USER/3d_graph/pipeline:/workspace \
 -B /cluster/work/cvg/students/nec:/shared_storage \
 -B /cluster/work/cvg/students/nec/.cache/huggingface:/root/.cache/huggingface \
 --pwd /workspace \
 --env MPLCONFIGDIR=/tmp/matplotlib \
 --env YOLO_CONFIG_DIR=/tmp/ultralytics \
 --env DISPLAY=${DISPLAY:-:0} \
 --env HF_HOME=/root/.cache/huggingface \
 --nv --containall \
 /cluster/work/cvg/students/nec/singularity/pipeline.sif \
 bash -c "mkdir -p /tmp/matplotlib /tmp/ultralytics && python pipeline.py"