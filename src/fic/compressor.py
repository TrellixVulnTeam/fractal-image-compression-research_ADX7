import itertools
import struct
from typing import List

import torch
from bitarray import bitarray

from .compressor_config import CompressorConfig
from .i_compressor import ICompressor
from .image_quad_tree import ImageQuadTree


class NaiveCompressor(ICompressor):
    def __init__(self, config: CompressorConfig):
        self._config: CompressorConfig = config

    def encode(self, image: torch.tensor) -> bytes:
        encoded: bitarray = bitarray()
        if image.shape[0] == 1:
            encoded.append(0)
        else:
            encoded.append(1)
        encoded.frombytes(struct.pack("ii", image.shape[1], image.shape[2]))
        tree: ImageQuadTree = ImageQuadTree(image, self._config)
        source_blocks: List[torch.tensor] = self._build_source_blocks(image)
        tree.build_tree(source_blocks)
        encoded += tree.encode()
        return encoded.tobytes()

    def decode(self, compressed: bytes) -> torch.tensor:
        encoded: bitarray = bitarray()
        encoded.frombytes(compressed)
        has_3_channels: bool = bool(encoded.pop(0))
        channels: int = 1
        if has_3_channels:
            channels = 3
        height, width = struct.unpack("ii", encoded[:64])
        del encoded[:64]
        image: torch.tensor = torch.randn((channels, height, width))
        tree: ImageQuadTree = ImageQuadTree(image, self._config)
        tree.decode(encoded)
        for i in range(50):
            source_blocks: List[torch.tensor] = self._build_source_blocks(tree._image.clone())
            tree.decode_image(source_blocks)
        return tree._image.clone()

    def _build_source_blocks(self, image: torch.tensor) -> List[torch.tensor]:
        blocks: List[torch.tensor] = []
        for i in range(self._config.source_block_levels):
            size: int = self._config.source_block_start_size // (2 ** i)
            step: int = max(int(size * self._config.source_block_step), 1)
            for x in range(0, image.shape[-1], step):
                for y in range(0, image.shape[-1], step):
                    to_x: int = x + size
                    to_y: int = y + size
                    new_part: torch.tensor = image[:, x:to_x, y:to_y]
                    if new_part.shape[-1] == size and new_part.shape[-2] == size:
                        blocks.append(new_part)
        return blocks
