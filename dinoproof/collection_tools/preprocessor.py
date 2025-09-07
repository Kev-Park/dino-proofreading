import torch
import torch.nn as nn

# Class for preprocessing image data (cleaning up, resizing, identifying points, augmenting dataset)

class DataPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def process(self, desired_size, output_directory):
        pass