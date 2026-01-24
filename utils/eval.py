import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from scipy import linalg
from tqdm import tqdm
import os

BASE_WORK_DIR = os.environ.get("BASE_WORK_DIR")
REAL_FEATURES_CACHE = os.path.join(BASE_WORK_DIR, "real_features.npy")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EPS = 1e-6

_feat_model = None
_prob_model = None


def get_inception_model(device="cuda", features_only=True):
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
    if features_only:
        model.fc = nn.Identity()
    model.aux_logits = False
    model.eval().to(device)
    return model

_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def preprocess_batch(x):
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return _normalize(x)


@torch.no_grad()
def get_metrics_data(images, batch_size=128, device="cuda"):
    global _prob_model
    if _prob_model is None:
        _prob_model = get_inception_model(device, features_only=False)

    features = []
    probs = []

    def get_features(module, input, output):
        features.append(output.view(output.size(0), -1).cpu().numpy())

    handle = _prob_model.avgpool.register_forward_hook(get_features)

    if isinstance(images, torch.Tensor):
        dataset = torch.utils.data.TensorDataset(images)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    else:
        loader = images

    try:
        for batch in tqdm(loader, desc="FID/IS Progress"):
            img_batch = batch[0] if isinstance(batch, (list, tuple)) else batch

            if img_batch.min() < 0: img_batch = (img_batch + 1) / 2
            img_batch = img_batch.clamp(0, 1).to(device)
            img_batch = preprocess_batch(img_batch)

            logits = _prob_model(img_batch)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(features, axis=0), np.concatenate(probs, axis=0)

def calculate_fid(real_features, fake_features):
    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((sigma1 + EPS * np.eye(sigma1.shape[0])) @ (sigma2 + EPS * np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def calculate_inception_score(probs, splits=10):
    N = probs.shape[0]
    scores = []
    for i in range(splits):
        part = probs[i * N // splits:(i + 1) * N // splits]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

def get_metric_scores(real_images, generated_images):
    if os.path.exists(REAL_FEATURES_CACHE):
        real_features = np.load(REAL_FEATURES_CACHE)
    else:
        real_features, _ = get_metrics_data(real_images)
        np.save(REAL_FEATURES_CACHE, real_features)
    fake_features, fake_probs = get_metrics_data(generated_images)
    fid = calculate_fid(real_features, fake_features)
    is_mean, is_std = calculate_inception_score(fake_probs)
    return {"FID": fid, "IS_MEAN": is_mean, "IS_STD": is_std}
