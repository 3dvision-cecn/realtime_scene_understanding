#!/usr/bin/bash

# Default username (can be overridden with -u flag)
CLUSTER_USER=${USER}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--user)
      CLUSTER_USER="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# SYNC local changes
ssh $CLUSTER_USER@euler mkdir -p /cluster/work/cvg/students/ndickenmann/3d_graph/pipeline
rsync -avzh --info=progress2 * $CLUSTER_USER@euler:/cluster/work/cvg/students/ndickenmann/3d_graph/pipeline --exclude .git/ \
--exclude docker/singularity.sif --exclude samples/01.zip --exclude yolo12x.pt \
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
bash "/cluster/work/cvg/students/ndickenmann/3d_graph/pipeline/docker/run_singularity.sh" "/cluster/work/cvg/students/ndickenmann/3d_graph/pipeline" "$2" "${@:3}"
EOT

ssh $CLUSTER_USER@euler sbatch < job.sh   
rm job.sh

