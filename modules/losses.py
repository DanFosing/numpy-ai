import numpy as np

def cross_entropy_loss(y, y_pred, eps=1e-9):
    return -np.mean(y * np.log(y_pred + eps))