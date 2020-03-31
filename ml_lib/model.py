from os.path import join
import torch
import numpy as np

from torch import nn

class Model(nn.module):
    """
    Base model class.
    """
    def __init__(self):
        """
        Constructor. Only calls the nn.module initialization
        """
        super().__init__()

    def save(self, name, checkpoint_dir):
        """
        Saves model
        
        Parameters:
            name: Name to save the model as
            checkpoint_dir: Directory in which to save the model
        """
        model_save_path = join(checkpoint_dir, f"{name}.pt")
        torch.save(self, model_save_path)
        print(f"Model saved to {model_save_path}")
