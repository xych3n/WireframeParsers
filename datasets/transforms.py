from typing import Any, Callable, Iterable, List, Tuple

from torch import Tensor
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: Iterable[Callable[..., Any]]):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor:
    def __call__(self, img, *args):
        img = F.to_tensor(img)
        return img, *args


class Resize:
    def __init__(self, size: int | Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img, ann = None, /):
        img = F.resize(img, self.size, antialias=True)
        return img, ann


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor: Tensor, *args):
        # img_tensor *= 255
        img_tensor = F.normalize(img_tensor, self.mean, self.std)
        return img_tensor, *args
