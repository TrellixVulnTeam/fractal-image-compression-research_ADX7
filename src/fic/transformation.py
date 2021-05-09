import struct
from typing import List

import torch
from bitarray import bitarray


class Transformation:
    def __init__(self, flip_v: bool, flip_h: bool, contrast: float = 1.0, brightness: float = 0.0):
        self._flip_v: bool = flip_v
        self._flip_h: bool = flip_h
        self._contrast: float = max(0.0, min(1.0, contrast))
        self._brightness: float = max(0.0, min(1.0, brightness))

    def __call__(self, src: torch.tensor) -> torch.tensor:
        flip_directions: List[int] = []
        if self._flip_v:
            flip_directions.append(-2)
        if self._flip_h:
            flip_directions.append(-1)
        flipped: torch.tensor = torch.flip(src, flip_directions)
        return self._contrast * flipped + self._brightness

    def encode(self) -> bitarray:
        encoded: bitarray = bitarray()
        encoded.append(self._flip_v)
        encoded.append(self._flip_h)
        encoded.frombytes(struct.pack("cc", bytes([int(self._contrast * 255)]), bytes([int(self._brightness * 255)])))
        return encoded

    def decode(self, encoded: bitarray):
        self._flip_v = bool(encoded.pop(0))
        self._flip_h = bool(encoded.pop(0))
        contrast, brightness = struct.unpack("cc", encoded[:16])
        self._contrast = int.from_bytes(contrast, byteorder="little") / 255
        self._brightness = int.from_bytes(brightness, byteorder="little") / 255
        del encoded[:16]
