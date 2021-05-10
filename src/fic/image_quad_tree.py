import struct
from typing import Optional, List, Tuple

import torch
from bitarray import bitarray
from torchvision.transforms import Resize

from .compressor_config import CompressorConfig
from .transformation import Transformation
from .transformation_searcher import TransformationSearcher


class ImageQuadTree:
    def __init__(self, image: torch.tensor, config: CompressorConfig):
        self._children: Optional[List[Optional[ImageQuadTree]]] = None
        self._image: Optional[torch.tensor] = image
        self._source_idx: Optional[int] = None
        self._transformation: Optional[Transformation] = None
        self._config: CompressorConfig = config
        self._transformation_searcher: TransformationSearcher = TransformationSearcher(config.loss_tolerance)

    def _split(self):
        assert self._children is None
        image_size: int = self._image.shape[-1]
        self._children = []
        for i in range(4):
            if i == 0:
                new_image_part: torch.tensor = self._image[:, :image_size // 2, :image_size // 2]
            elif i == 1:
                new_image_part: torch.tensor = self._image[:, :image_size // 2, image_size // 2:]
            elif i == 2:
                new_image_part: torch.tensor = self._image[:, image_size // 2:, :image_size // 2]
            else:
                new_image_part: torch.tensor = self._image[:, image_size // 2:, image_size // 2:]
            new_child_tree: ImageQuadTree = ImageQuadTree(new_image_part, self._config)
            self._children.append(new_child_tree)

    def build_tree(self, source_blocks: List[torch.tensor]):
        image_size: int = self._image.shape[-1]
        if image_size >= self._config.source_block_start_size:
            self._split()
            for node in self._children:
                node.build_tree(source_blocks)
        else:
            source_idx, transformation, loss = self._find_the_best_source_block(source_blocks)
            if loss <= self._config.loss_tolerance:
                self._transformation = transformation
                self._source_idx = source_idx
            elif image_size > self._config.min_destination_block_size:
                self._split()
                for node in self._children:
                    node.build_tree(source_blocks)
            else:
                self._transformation = transformation
                self._source_idx = source_idx
                print(loss)

    def _find_the_best_source_block(self, source_blocks: List[torch.tensor]) -> Tuple[int, Transformation, float]:
        best_transformation: Transformation = Transformation(False, False, 1.0, 0.0)
        best_loss: torch.tensor = torch.tensor(float("inf"), dtype=torch.float32)
        best_source_idx: int = 0
        for source_idx, source in enumerate(source_blocks):
            if source.shape[-1] <= self._image.shape[-1]:
                continue
            transformation, loss = self._transformation_searcher(source, self._image)
            if loss < best_loss:
                best_loss = loss
                best_transformation = transformation
                best_source_idx = source_idx
            if best_loss <= self._config.loss_tolerance:
                return best_source_idx, best_transformation, best_loss
        return best_source_idx, best_transformation, best_loss

    def encode(self) -> bitarray:
        encoded: bitarray = bitarray()
        queue: List[ImageQuadTree] = []
        transformations: List[(int, Transformation)] = []
        for child in self._children:
            queue.append(child)
        while queue:
            current_tree: ImageQuadTree = queue.pop(0)
            if current_tree._children is not None:
                encoded.append(True)
                for child in current_tree._children:
                    queue.append(child)
            else:
                encoded.append(False)
                transformations.append((current_tree._source_idx, current_tree._transformation))
        for transformation in transformations:
            encoded.frombytes(struct.pack("i", transformation[0]))
            encoded += transformation[1].encode()
        return encoded

    def decode(self, encoded: bitarray):
        self._split()
        queue: List[ImageQuadTree] = []
        leafs: List[ImageQuadTree] = []
        for child in self._children:
            queue.append(child)
        while queue:
            current_tree: ImageQuadTree = queue.pop(0)
            is_splitted: bool = bool(encoded.pop(0))
            if is_splitted:
                current_tree._split()
                queue += current_tree._children
            else:
                leafs.append(current_tree)
        for leaf in leafs:
            leaf._source_idx = struct.unpack("i", encoded[:32])[0]
            del encoded[:32]
            leaf._transformation = Transformation(False, False)
            leaf._transformation.decode(encoded)

    def decode_image(self, source_blocks: List[torch.tensor]):
        queue: List[ImageQuadTree] = [self]
        while queue:
            current_tree: ImageQuadTree = queue.pop(0)
            if current_tree._children is None:
                source: torch.tensor = Resize(current_tree._image.shape[-1])(source_blocks[current_tree._source_idx])
                updated_part: torch.tensor = current_tree._transformation(source)
                current_tree._image.copy_(updated_part)
            else:
                queue += current_tree._children
