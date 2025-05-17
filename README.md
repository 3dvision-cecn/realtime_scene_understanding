1. create a conda env and install the 
2. sync your dropbox checpoints with the conf/checkpoints
3. sync recordings ln -s /home/usert/dropbox/recordings/ recordings
4. change the paths in the conf file
```
ln -s /home/$USER/dropbox/checkpoints/ conf/checkpoints
conda env create --name 3d_graph --file=environment.yml
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


To stream rerun to local:

1. run
```
#!/usr/bin/env bash

# Forward remote ports 9090 (HTTP) and 9877 (WebSocket) back to your laptop:
ssh -i ~/.ssh/id_ed25519 -p 10351 \
    -L 9090:localhost:9090 \
    -L 9876:localhost:9876 \
    root@213.173.109.196
```

2. run
```
 rerun --connect rerun+http://localhost:9876/proxy
```
