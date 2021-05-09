import os
from typing import Callable

from torchvision.transforms import Compose, Resize, Grayscale, CenterCrop, ToPILImage, Normalize

from fic import ImageDataset
from fic.compressor import NaiveCompressor
from fic.compressor_config import CompressorConfig


def main():
    transforms: Callable = Compose([
        Resize(128),
        CenterCrop(128),
        Grayscale(),
    ])
    dataset: ImageDataset = ImageDataset(os.path.join("data", "imagenette2-320", "train"), transforms)
    config = CompressorConfig()
    compressor: NaiveCompressor = NaiveCompressor(config)
    b = compressor.encode(dataset[10])
    print(len(b))
    ToPILImage()(dataset[10]).save("test.jpg", "JPEG")
    ToPILImage()(dataset[10]).save("test.png", "PNG")
    img = compressor.decode(b)
    ToPILImage()(img).save("fr.png", "PNG")


if __name__ == '__main__':
    main()
