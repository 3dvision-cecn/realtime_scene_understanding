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
```
# Download Sample iPhone data


```

# How to Run Docker
```
# some of the repos might be private build might not work
docker build --ssh default=${SSH_AUTH_SOCK} -t pipeline:latest docker/
# directly pull from docker io
docker pull nemantor31/pipeline
# run the pipeline with the iPhone dataset and get rerun visualisations
./docker/run_docker.bash
```