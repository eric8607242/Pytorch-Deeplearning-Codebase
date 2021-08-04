import os
import os.path as osp
import random

import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset

def get_data_dataloader(dataflow_config, *args, **kwargs):
    dataset_path = dataflow_config["dataset_path"]
    input_size = dataflow_config["input_size"]
    batch_size = dataflow_config["batch_size"]
    num_workers = dataflow_config["num_workers"]
    train_portion = dataflow_config["train_portion"]

    return train_loader, val_loader, test_loader

def get_train_val_dataset(dataset, train_portion):
    dataset_len = len(dataset)

    train_dataset_len = int(dataset_len * train_portion)
    val_dataset_len = int(dataset_len * (1 - train_portion))

    #val_start_index = random.randrange(train_dataset_len)
    val_start_index = 100
    indices = torch.arange(dataset_len)

    train_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_dataset_len:]])
    val_indices = indices[val_start_index:val_start_index+val_dataset_len]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset
