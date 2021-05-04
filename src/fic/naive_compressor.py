import itertools
import struct
from typing import List, Tuple

import torch
import torchvision
from bitarray import bitarray
from tqdm.auto import tqdm

from .i_compressor import ICompressor
from .naive_compressor_config import NaiveCompressorConfig


class NaiveCompressor(ICompressor):
    def __init__(self, config: NaiveCompressorConfig):
        self._config: NaiveCompressorConfig = config
        self._vertical_flips_variants: List[bool] = [True, False]
        self._horizontal_flips_variants: List[bool] = [True, False]
        self._criterion: torch.nn.Module = torch.nn.MSELoss()
        self._resizer: torchvision.transforms.Resize = torchvision.transforms.Resize(config.destination_block_size)

    @staticmethod
    def _get_image_parts(image: torch.tensor, part_size: int) -> torch.tensor:
        total_parts: int = image.shape[-1] // part_size
        for x, y in itertools.product(range(total_parts), range(total_parts)):
            yield image[:, x * part_size:x * part_size + part_size, y * part_size:y * part_size + part_size]

    def encode(self, image: torch.tensor) -> bytes:
        source_size: int = self._config.source_block_size
        destination_size: int = self._config.destination_block_size
        encoded: bitarray = bitarray()
        total_parts: int = image.shape[-1] // destination_size
        for destination in tqdm(NaiveCompressor._get_image_parts(image, destination_size),
                                total=total_parts * total_parts):
            best_loss: torch.tensor = torch.tensor(float("inf"), dtype=torch.float32)
            best_transform: Tuple[bool, bool, float, float] = (False, False, 1, 0)
            for source in NaiveCompressor._get_image_parts(image, source_size):
                transform, loss = self._find_the_best_transform(source, destination)
                if loss < best_loss:
                    best_loss = loss
                    best_transform = transform
            encoded.append(best_transform[0])
            encoded.append(best_transform[1])
            encoded.frombytes(struct.pack("f", best_transform[2]))
            encoded.frombytes(struct.pack("f", best_transform[3]))
        return encoded.tobytes()

    def decode(self, compressed: bytes) -> torch.tensor:
        pass

    def _transform(self, src: torch.tensor, vertical_flip: bool, horizontal_flip: bool, contrast: float = 1.0,
                   brightness: float = 0) -> torch.tensor:
        reduced: torch.tensor = self._resizer(src)
        flip_directions: List[int] = []
        if vertical_flip:
            flip_directions.append(-2)
        if horizontal_flip:
            flip_directions.append(-1)
        flipped: torch.tensor = torch.flip(reduced, flip_directions)
        return contrast * flipped + brightness

    def _find_the_best_transform(self, src: torch.tensor, dst: torch.tensor) -> Tuple[
        Tuple[bool, bool, float, float], float]:
        best_transform: Tuple[bool, bool, float, float] = (False, False, 1, 0)
        ones: torch.tensor = torch.ones((dst.shape[-1] * dst.shape[-2] * dst.shape[-3], 1))
        best_loss: torch.tensor = torch.tensor(float("inf"), dtype=torch.float32)
        dst_r = dst.reshape(-1, 1)
        for horizontal_flip in self._horizontal_flips_variants:
            for vertical_flip in self._vertical_flips_variants:
                test_transform: torch.tensor = self._transform(src, vertical_flip, horizontal_flip)
                flattened_test_transforms: torch.tensor = test_transform.view(-1, 1)
                test_transform_a: torch.tensor = torch.cat([flattened_test_transforms, ones], dim=1)
                solution, _ = torch.lstsq(dst_r, test_transform_a)
                contrast: float = solution[0].item()
                brightness: float = solution[1].item()
                test_transform = contrast * test_transform + brightness
                loss: torch.tensor = self._criterion(test_transform, dst)
                if loss < best_loss:
                    best_loss = loss
                    best_transform = (vertical_flip, horizontal_flip, contrast, brightness)
        return best_transform, best_loss.item()
