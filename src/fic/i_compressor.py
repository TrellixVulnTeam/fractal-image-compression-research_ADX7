from abc import ABC, abstractmethod

import torch


class ICompressor(ABC):
    @abstractmethod
    def encode(self, image: torch.tensor) -> bytes:
        raise NotImplemented()

    @abstractmethod
    def decode(self, compressed: bytes) -> torch.tensor:
        raise NotImplemented()
