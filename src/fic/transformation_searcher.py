from typing import Tuple

import torch
from torchvision.transforms import Resize

from .transformation import Transformation


class TransformationSearcher:
    def __init__(self, loss_tolerance: float):
        self._criterion: torch.nn.Module = torch.nn.MSELoss()
        self._loss_tolerance: float = loss_tolerance

    def __call__(self, src: torch.tensor, dst: torch.tensor) -> Tuple[Transformation, float]:
        resize: Resize = Resize(dst.shape[-1])
        src = resize(src)
        best_transformation: Transformation = Transformation(False, False, 1.0, 0.0)
        best_loss: torch.tensor = torch.tensor(float("inf"), dtype=torch.float32)
        ones: torch.tensor = torch.ones((dst.shape[-1] * dst.shape[-2] * dst.shape[-3], 1))
        dst_r = dst.reshape(-1, 1)
        for horizontal_flip in [False, True]:
            for vertical_flip in [False, True]:
                transformation: Transformation = Transformation(vertical_flip, horizontal_flip, 1.0, 0.0)
                transformation_result: torch.tensor = transformation(src)
                flattened_transformation_result: torch.tensor = transformation_result.view(-1, 1)
                test_transform_a: torch.tensor = torch.cat([flattened_transformation_result, ones], dim=1)
                try:
                    solution, _ = torch.lstsq(dst_r, test_transform_a)
                except RuntimeError:
                    continue
                contrast: float = solution[0].item()
                brightness: float = solution[1].item()
                new_transformation: Transformation = Transformation(vertical_flip, horizontal_flip, contrast,
                                                                    brightness)
                transformation_result = new_transformation(src)
                loss: torch.tensor = self._criterion(transformation_result, dst)
                if loss < best_loss:
                    best_loss = loss
                    best_transformation = new_transformation
                if best_loss <= self._loss_tolerance:
                    return best_transformation, best_loss.item()
        return best_transformation, best_loss.item()
