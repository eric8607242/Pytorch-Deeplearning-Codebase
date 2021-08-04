import sys

def get_model_class(name):
    return getattr(sys.modules[__name__], f"Model")

from .model import Model
