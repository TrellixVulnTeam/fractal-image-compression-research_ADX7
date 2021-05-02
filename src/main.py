import os
from typing import Callable

import torch
from torchvision.transforms import Compose, Resize, Grayscale, CenterCrop

from fic import ImageDataset


def main():
    transforms: Callable = Compose([
        Resize(256),
        CenterCrop(256),
        Grayscale()
    ])
    dataset: ImageDataset = ImageDataset(os.path.join("data", "imagenette2-320", "train"), transforms)
    print(torch.tensor(float("inf"), dtype=torch.float32))


if __name__ == '__main__':
    main()
