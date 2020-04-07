import torch
from torch.nn import MSELoss

def r2(outputs, labels):
    """
    Calculates r2 score for a set of outputs and labels. Ranges from -infinity
    to 1, with values closer to 1 being better

    Parameters:
        outputs: tensor of shape (N, *), and is the output being scored
        labels: tensor of shape (N, *), the labels being scored against
    """
    mse_loss = MSELoss()

    mean = torch.mean(labels)
    mean_var = mse_loss(mean, labels)
    outputs_var = mse_loss(outputs, labels)
    r2 = 1 - (outputs_var / mean_var)
    return r2
    
    
