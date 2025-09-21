#! /usr/bin/bash

# Get CLUSTER_USER from command line argument
CLUSTER_USER=${2:-$USER}

# Set environment variables for proper config directory handling
export MPLCONFIGDIR=/tmp/matplotlib
export YOLO_CONFIG_DIR=/tmp/ultralytics
export DISPLAY=${DISPLAY:-:0}

# load modules
module load eth_proxy

# Copy files to $SCRATCH for speed
echo "Copying shared storage to $SCRATCH..."
cp -r /cluster/work/cvg/students/nec $SCRATCH/shared_storage

singularity exec \
 -B /cluster/work/cvg/students/$CLUSTER_USER/3d_graph/pipeline:/workspace \
 -B $SCRATCH/shared_storage:/shared_storage \
 -B $SCRATCH/shared_storage/.cache2/huggingface:/root/.cache/huggingface \
 --pwd /workspace \
 --env MPLCONFIGDIR=/tmp/matplotlib \
 --env YOLO_CONFIG_DIR=/tmp/ultralytics \
 --env DISPLAY=${DISPLAY:-:0} \
 --env HF_HOME=/root/.cache/huggingface \
 --nv --containall \
 /cluster/work/cvg/students/nec/singularity/pipeline.sif \
 bash -c "mkdir -p /tmp/matplotlib /tmp/ultralytics && rm -rf /workspace/conf/checkpoints && ln -s /shared_storage/checkpoints/checkpoints conf/checkpoints && ls conf/checkpoints/ && python pipeline.py --full"