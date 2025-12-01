"""
Minimal version of common.py for inference only
Original file backed up as common_original.py
"""
import torch
import torch.nn as nn

def initialize_weights(*models):
    """Initialize model weights"""
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

# Stub for training functions that won't be used
class TrainCollect:
    """Stub class for training - not used in inference"""
    pass
