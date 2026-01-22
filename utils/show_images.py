import torchvision
import matplotlib.pyplot as plt
import torch

def show_images(images, nrow=8, figsize=(8,8)):

    if images.min() < 0:
        print("Scale Required")
        images = (images * 0.5) + 0.5
    images = torch.clamp(images, 0, 1)

    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=figsize)
    plt.imshow(np_grid)
    plt.axis('off')
    plt.show()