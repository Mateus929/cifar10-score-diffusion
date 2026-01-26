def dsm_loss(score_noisy, epsilon, sigma, lambda_t):
    score_target = -epsilon / sigma
    diff = score_noisy - score_target
    loss = 0.5 * (diff ** 2).sum(dim=(1, 2, 3))
    return (lambda_t * loss).mean()


import torch

def dsm_loss_fixed_lambda(score_pred, epsilon, sigma):
    loss = 0.5 * torch.mean(torch.sum((score_pred * sigma + epsilon) ** 2, dim=[1, 2, 3]))
    return loss

def ncsn_loss(score_noisy, epsilon, sigma):
    diff = sigma * score_noisy + epsilon
    loss = 0.5 * (diff ** 2).mean(dim=(1, 2, 3))
    return loss.mean()
