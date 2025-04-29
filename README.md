1. create a conda enc and install the 
2. syns your dropbox checpoints with the conf/checkpoints
3. change the paths in the conf file
```
ln -s /home/$USER/dropbox/checkpoints/ conf/checkpoints
conda env create --name 3d_graph --file=environments.yml
```