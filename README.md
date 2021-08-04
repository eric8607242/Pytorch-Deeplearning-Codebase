# Pytorch-Deeplearning-Codebase
Pytorch-Deeplearning-Codebase provides a flexible codebase for constructing the various deep learning training pipeline efficiently.

## Usage
```
python3 main.py -c [CONFIG FILE] --title [EXPERIMENT TITLE]

optional arguments:
    --title                 The title of the experiment. All corrsponding files will be saved in the directory named with experiment title.
    -c, --config            The path to the config file. Refer to ./config/ for serveral example config file.
```

## Features
1. Training information logging.
    - In Pytorch-Deeplearning-Codebase, users can record the training information with the loggers provied in default (e.g., Logger and TensorboardX).
2. Modularize main components for deep learning training pipeline.
    - Pytorch-Deeplearning-Codebase modulizes the main components of training pipeline (e.g., model, dataflow, criterion, and trainer).
