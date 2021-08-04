import sys

def get_dataloader(dataset_name, config):
    dataloader_builder = getattr(sys.modules[__name__], f"get_{dataset_name}_dataloader")

    return dataloader_builder(config)


# Import customizing module
from .dataflow import get_data_dataloader
