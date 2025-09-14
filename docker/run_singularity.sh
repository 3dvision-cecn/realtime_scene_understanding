#! /usr/bin/bash

singularity exec -B /cluster/scratch/$USER/3d_graph/pipeline:/workspace \
 -B /cluster/work/cvg/students/nec:/shared_storage \
 --nv --containall /cluster/work/cvg/students/nec/singularity/pipeline.sif \
 bash -c "python /workspace/pipeline.py"