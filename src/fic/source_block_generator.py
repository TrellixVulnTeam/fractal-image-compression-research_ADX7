from typing import Generator

import torch


class SourceBlockGenerator:
    def __init__(self, source_block_levels: int, source_block_start_size: int, source_block_step: float):
        self._source_block_levels: int = source_block_levels
        self._source_block_start_size: int = source_block_start_size
        self._source_block_step: float = source_block_step

    def __call__(self, image: torch.tensor) -> Generator[torch.tensor, None, None]:
        for i in range(self._source_block_levels):
            size: int = self._source_block_start_size // (2 ** i)
            step: int = max(int(size * self._source_block_step), 1)
            for x in range(0, image.shape[-1], step):
                for y in range(0, image.shape[-1], step):
                    to_x: int = x + size
                    to_y: int = y + size
                    new_part: torch.tensor = image[:, x:to_x, y:to_y]
                    if new_part.shape[-1] == size and new_part.shape[-2] == size:
                        yield new_part
