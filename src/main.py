import os
from typing import Callable

from fic import ImageDataset
from torchvision.transforms import Compose, Resize, Grayscale, CenterCrop, ToPILImage


def main():
    transforms: Callable = Compose([
        Resize(256),
        CenterCrop(256),
        Grayscale()
    ])
    dataset: ImageDataset = ImageDataset(os.path.join("data", "imagenette2-320", "train"), transforms)




if __name__ == '__main__':
    main()
