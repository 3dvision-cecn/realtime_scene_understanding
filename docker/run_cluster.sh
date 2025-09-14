#!/usr/bin/bash

# SYNC localc changes
ssh euler mkdir -p /cluster/scratch/$USER/3d_graph/pipeline/
rsync -avzh --info=progress2 * euler://cluster/scratch/$USER/3d_graph/pipeline/  --exclude .git/ \
--exclude docker/singularity.sif --exclude samples/01.zip --exclude yolo12x.pt --exclude third_party/ \
--exclude docker/singularity.sif.tar --exclude docker/singularity.sif \
--exclude docker/singularity.sif.tar


cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "/cluster/scratch/$USER/3d_graph/pipeline/docker/run_singularity.sh" "/cluster/scratch/$USER/3d_graph/pipeline" "$2" "${@:3}"
EOT

ssh euler sbatch < job.sh   
rm job.sh

