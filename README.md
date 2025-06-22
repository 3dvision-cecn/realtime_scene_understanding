# Reltime Scene understanding

# project Structure
```
conf/
- config.yaml 
- checkpoints/
    - avion/
    - edgetam/
    - hamer_ckpts/
    - omni_dc/
dataset/
- dataset/
    - recordings/ # iPhone recordings unzipped red folders
    - HD-EPIC/    # hd-epic 

```

# How to Download the Dataset
Downloading the sample iPhone data
```
# Download Sample iPhone data
git lfs install
git pull
# unzip and put into dataset/recordings/train/01
```
Downloading HD-EPIC
```
# use the faster pipeline/utils/hd-epic-downloader.py for parallel downloading
python pipeline/utils/hd-epic-downloader.py dataset --vrs --slam-gaze --participants 1
```

# How to Run Docker
```
# some of the repos might be private build might not work
docker build --ssh default=${SSH_AUTH_SOCK} -t pipeline:latest docker/
# directly pull from docker io
docker pull nemantor31/pipeline
# run the pipeline with the iPhone dataset and get rerun visualisations
./docker/run_docker_iphone.bash
# run the pipeline with the HD-EPIC dataset and get rerun visualisations
./docker/run_docker_epic.bash
```

# Training the EK-100 Action Recognition Network
```
# run the full processing on the data and get hf5 files
python training/training.py --dataset_dir=????
```