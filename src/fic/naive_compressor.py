from typing import List, Tuple

import torch
import torchvision

from i_compressor import ICompressor
from naive_compressor_config import NaiveCompressorConfig


class NaiveCompressor(ICompressor):
    def __init__(self, config: NaiveCompressorConfig):
        self._config: NaiveCompressorConfig = config
        self._vertical_flips_variants: List[bool] = [True, False]
        self._horizontal_flips_variants: List[bool] = [True, False]
        self._criterion: torch.nn.Module = torch.nn.MSELoss()

    def encode(self, image: torch.tensor) -> bytes:
        # parallelize it!
        pass

    def decode(self, compressed: bytes) -> torch.tensor:
        pass

    def _transform(self, src: torch.tensor, vertical_flip: bool, horizontal_flip: bool, contrast: float = 1.0,
                   brightness: float = 0) -> torch.tensor:
        reduced: torch.tensor = torchvision.transforms.Resize(self._config.destination_block_size)(src)
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
        ones: torch.tensor = torch.ones((dst.shape[-1] * dst.shape[-2], 1))
        best_loss: torch.tensor = torch.tensor(float("inf"), dtype=torch.float32)
        for horizontal_flip in self._horizontal_flips_variants:
            for vertical_flip in self._vertical_flips_variants:
                test_transform: torch.tensor = self._transform(src, vertical_flip, horizontal_flip)
                flattened_test_transforms: torch.tensor = test_transform.view(-1, 1)
                test_transform_a: torch.tensor = torch.stack([flattened_test_transforms, ones], dim=1)
                solution, _ = torch.lstsq(test_transform_a, dst.view(-1, 1))
                contrast: float = solution[0].item()
                brightness: float = solution[1].item()
                test_transform = contrast * test_transform + brightness
                loss: torch.tensor = self._criterion(test_transform, dst)
                if loss < best_loss:
                    best_loss = loss
                    best_transform = (vertical_flip, horizontal_flip, contrast, brightness)
        return best_transform, best_loss.item()
