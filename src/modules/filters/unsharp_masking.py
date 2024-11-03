from typing import Optional

from torch import Tensor

from modules.filters.base import Filter
from modules.filters.gauisian import GaussianFilter


class UnsharpMaskFilter(Filter):
    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 1.0,
        amount: float = 1.5,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__(kernel_size, stride, padding)
        self.sigma = sigma
        self.amount = amount
        self.gaussian_filter = GaussianFilter(kernel_size, sigma, stride)

    def apply(self, image: Tensor) -> Tensor:
        blurred_image = self.gaussian_filter.apply(image)
        high_pass = image - blurred_image
        sharpened_image = image + self.amount * high_pass
        return sharpened_image
