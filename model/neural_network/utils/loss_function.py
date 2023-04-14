import torch
import numpy as np
import torch.nn as nn


# RMSLE(Root Mean Squared Log Error)
class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.mse(torch.log(input + 1), torch.log(target + 1)))
