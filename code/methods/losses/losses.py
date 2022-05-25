import torch
import numpy as np


def kl_loss(mu, logvar):

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kl_divergence


def log_barrier(z, t=5):

    # Only one value
    if z.shape[0] == 1:

        if z <= - 1 / t ** 2:
            log_barrier_loss = - torch.log(-z) / t
        else:
            log_barrier_loss = t * z + -np.log(1 / (t ** 2)) / t + 1 / t

    # Constrain over multiple values
    else:
        log_barrier_loss = torch.tensor(0).cuda().float()
        for i in np.arange(0, z.shape[0]):
            zi = z[i, 0]
            if zi <= - 1 / t ** 2:
                log_barrier_loss += - torch.log(-zi) / t
            else:
                log_barrier_loss += t * zi + -np.log(1 / (t ** 2)) / t + 1 / t

    return log_barrier_loss
