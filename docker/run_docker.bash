#!/usr/bin/bash
docker run -t --gpus=all --rm -v $PWD:/workspace \
                         -v $PWD/dataset:/workspace/dataset \
                         -v $PWD/conf/checkpoints:/workspace/conf/checkpoints  \
                        nemantor31/pipeline:latest python pipeline.py --dataset=hd_epic --full