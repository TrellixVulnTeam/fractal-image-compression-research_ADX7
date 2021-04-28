import torch

from i_compressor import ICompressor
from naive_compressor_config import NaiveCompressorConfig


class NaiveCompressor(ICompressor):
    def __init__(self, config: NaiveCompressorConfig):
        self._config: NaiveCompressorConfig = config

    def encode(self, image: torch.tensor) -> bytes:
        pass

    def decode(self, compressed: bytes) -> torch.tensor:
        pass

    def _transform(self, src: torch.tensor, direction: int, angle: int, contract: float = 1.0, brightness: float = 0):
        pass
