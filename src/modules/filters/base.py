from torch import Tensor
from typing import Optional

from utils.conv import torch_conv


class Filter:
    def __init__(
        self, kernel_size: int, stride: int = 1, padding: Optional[int] = None
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2

    def create_kernel(self) -> Tensor:
        raise NotImplementedError

    def apply(self, image: Tensor) -> Tensor:
        kernel = self.create_kernel()
        return torch_conv(image=image, kernel=kernel, stride=self.stride, padding=self.padding)
