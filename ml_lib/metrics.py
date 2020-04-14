import numpy as np
import torch
from torch.nn import MSELoss
from sklearn.metrics import r2_score, roc_auc_score

METRIC_FUNCS = dict()

def register(name):
    """
    Decorator to store metric functions into METRIC dict for easy lookup
    """
    def register_decorator(func):
        METRIC_FUNCS[name] = func
        return func
    return register_decorator

@register("r2")
def calc_r2(labels, outputs):
    """
    Calculates r2 score for a set of outputs and labels. Ranges from -infinity
    to 1, with values closer to 1 being better

    Parameters:
        outputs: tensor of shape (N, *), and is the output being scored
        labels: tensor of shape (N, *), the labels being scored against
    """
    # mse_loss = MSELoss()
    # mean = torch.mean(labels)
    # mean_var = mse_loss(mean, labels)
    # outputs_var = mse_loss(outputs, labels.float())
    # r2 = 1 - (outputs_var / mean_var)
    # return r2.item()
    return r2_score(labels, outputs.detach())

@register("Binary Accuracy")
def calc_binary_accuracy(labels, outputs, thresh=0.5):
    """
    Calculates accuracy score for a set of outputs and labels.

    Parameters:
        outputs: tensor of shape (N, *), and is the output being scored
        labels: tensor of shape (N, *), the labels being scored against
        thresh: float, the threshhold to decide positive or negative with
    """
    prediction = (outputs > thresh).float()
    labels = labels.float().view_as(prediction)
    correct = prediction.eq(labels)
    correct = np.squeeze(correct.numpy())
    num_correct = np.sum(correct)
    num_pred = outputs.size(0)
    return num_correct / num_pred

@register("AUROC")
def calc_auroc(labels, outputs):
    """
    Calculates auroc score for a set of outputs and labels.

    Parameters:
        outputs: tensor of shape (N, *), and is the output being scored
        labels: tensor of shape (N, *), the labels being scored against
    """
    return roc_auc_score(labels, outputs.detach())

if __name__ == "__main__":
    print(METRICS)
