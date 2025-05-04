1. create a conda enc and install the 
2. syns your dropbox checpoints with the conf/checkpoints
3. sync recordings ln -s /home/usert/dropbox/recordings/ recordings
4. change the paths in the conf file
```
ln -s /home/$USER/dropbox/checkpoints/ conf/checkpoints
conda env create --name 3d_graph --file=environments.yml
```


Poetry for the win:

1. download poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. add poetry to the path
```
export PATH="$HOME/.local/bin:$PATH"
```
3. install packages
```
poetry install --no-root
```


4. run it
```
poetry run python pipeline.py
```

5. you might still need to install ffmpeg on your machine, also the clip repo needs to be in third party and the checkpoints configured as:
```
ln -s /home/$USER/dropbox/checkpoints/ conf/checkpoints
```
```
apt-get update
apt-get install -y ffmpeg
```