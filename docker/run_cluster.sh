#!/usr/bin/bash

# Default username (can be overridden with -u flag)
CLUSTER_USER=${USER}
SYNC_CACHE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--user)
      CLUSTER_USER="$2"
      shift 2
      ;;
    --no-cache)
      SYNC_CACHE=false
      shift
      ;;
    *)
      break
      ;;
  esac
done

# SYNC local changes
ssh $CLUSTER_USER@euler mkdir -p /cluster/work/cvg/students/$CLUSTER_USER/3d_graph/pipeline
rsync -avzh --info=progress2 \
--exclude '.git/' \
--exclude 'docker/singularity.sif' \
--exclude 'docker/singularity.sif.tar' \
--exclude 'samples/01.zip' \
--exclude 'temp/*' \
* \
$CLUSTER_USER@euler:/cluster/work/cvg/students/$CLUSTER_USER/3d_graph/pipeline

# SYNC HuggingFace cache for EdgeTAM RepViT model (optional)
if [ "$SYNC_CACHE" = true ]; then
  if [ -d ~/.cache/huggingface ]; then
    ssh $CLUSTER_USER@euler mkdir -p /cluster/work/cvg/students/nec/.cache/huggingface
    rsync -avzh --info=progress2 ~/.cache/huggingface/ $CLUSTER_USER@euler:/cluster/work/cvg/students/nec/.cache/huggingface/
  else
    echo "Warning: ~/.cache/huggingface not found, skipping cache sync"
  fi
else
  echo "Skipping HuggingFace cache sync (use without --no-cache to enable)"
fi


cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --account=es_hutter
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "/cluster/work/cvg/students/$CLUSTER_USER/3d_graph/pipeline/docker/run_singularity.sh" "/cluster/work/cvg/students/$CLUSTER_USER/3d_graph/pipeline" "$CLUSTER_USER" "$2" "${@:3}"
EOT

ssh $CLUSTER_USER@euler sbatch < job.sh   
rm job.sh

