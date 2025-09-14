#! /usr/bin/bash

# Set environment variables for proper config directory handling
export MPLCONFIGDIR=/tmp/matplotlib
export YOLO_CONFIG_DIR=/tmp/ultralytics
export DISPLAY=${DISPLAY:-:0}

singularity exec \
 -B /cluster/work/cvg/students/ndickenmann/3d_graph/pipeline:/workspace \
 -B /cluster/work/cvg/students/nec:/shared_storage \
 --pwd /workspace \
 --env MPLCONFIGDIR=/tmp/matplotlib \
 --env YOLO_CONFIG_DIR=/tmp/ultralytics \
 --env DISPLAY=${DISPLAY:-:0} \
 --nv --containall \
 /cluster/work/cvg/students/nec/singularity/pipeline.sif \
 bash -c "mkdir -p /tmp/matplotlib /tmp/ultralytics && python pipeline.py"