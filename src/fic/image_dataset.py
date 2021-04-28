import os
from glob import glob
from typing import Callable, Optional, List

from PIL import Image
from torch import tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


class ImageDataset(Dataset):
    def __init__(self, path_to_dataset: str, transforms: Optional[Callable] = None):
        self._path_to_images: List[str] = glob(os.path.join(path_to_dataset, "**", "*"))
        self._transforms: Optional[Callable] = transforms
        self._to_tensor_transform: Optional[Callable] = ToTensor()

    def __len__(self):
        return len(self._path_to_images)

    def __getitem__(self, index: int) -> tensor:
        path_to_image: str = self._path_to_images[index]
        with Image.open(path_to_image) as image:
            image_t: tensor = self._to_tensor_transform(image)
            if self._transforms is not None:
                image_t = self._transforms(image_t)
            return image_t
