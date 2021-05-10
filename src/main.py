import os
from typing import Callable

import torch
from torchvision.transforms import Compose, Resize, Grayscale, CenterCrop, ToPILImage

from fic import ImageDataset
from fic.compressor import NaiveCompressor
from fic.compressor_config import CompressorConfig


def main():
    transforms: Callable = Compose([
        Resize(64),
        CenterCrop(64),
        Grayscale(),
    ])
    dataset: ImageDataset = ImageDataset(os.path.join("data", "imagenette2-320", "train"), transforms)
    image: torch.tensor = dataset[1223]
    ToPILImage()(image).save("test.jpg", "JPEG")
    ToPILImage()(image).save("test.png", "PNG")
    config = CompressorConfig()
    compressor: NaiveCompressor = NaiveCompressor(config)
    with open("test.fic", "wb") as f:
        f.write(compressor.encode(image))
    with open("test.fic", "rb") as f:
        ToPILImage()(compressor.decode(f.read())).save("test.fic.png", "PNG")


if __name__ == '__main__':
    main()
