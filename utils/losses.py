def dsm_loss(score_noisy, epsilon, sigma, lambda_t):
    score_target = -epsilon / (sigma ** 2)
    diff = score_noisy - score_target
    loss = 0.5 * (diff ** 2).sum(dim=(1, 2, 3))
    return (lambda_t * loss).mean()
