import torch
from torch import Tensor

from modules.filters.base import Filter


class BoxFilter(Filter):
    def create_kernel(self) -> Tensor:
        kernel = torch.ones((self.kernel_size, self.kernel_size), dtype=torch.float32)
        kernel /= kernel.numel()
        return kernel
