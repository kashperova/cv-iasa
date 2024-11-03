import torch
from torch import Tensor

from modules.filters.base import Filter


class GaussianFilter(Filter):
    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 1.0,
        stride: int = 1
    ):
        padding = kernel_size // 2
        super().__init__(kernel_size, stride, padding)
        self.sigma = sigma

    def create_kernel(self) -> Tensor:
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel /= kernel.sum()
        return kernel
