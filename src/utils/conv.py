import numpy as np
from typing import Optional

from torch import Tensor
import torch.nn.functional as F


def numpy_conv(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: Optional[int] = 1,
    padding: Optional[int] = 0,
) -> np.ndarray:
    # add zeros padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # get size of image & kernel
    in_height, in_width = image.shape
    kernel_size = kernel.shape[0]

    out_height = (in_height - kernel_size) // stride + 1
    out_width = (in_width - kernel_size) // stride + 1

    output = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size

            image_patch = image[start_i:end_i, start_j:end_j]
            output[i, j] = np.sum(image_patch * kernel)

    return output


def torch_conv(
    image: Tensor,
    kernel: Tensor,
    stride: Optional[int] = 1,
    padding: Optional[int] = 0,
    groups: Optional[int] = 3
) -> Tensor:
    kernel_size = kernel.shape[0]
    kernel = kernel.expand(3, 1, kernel_size, kernel_size)
    output = F.conv2d(image, kernel, stride=stride, padding=padding, groups=groups)

    return output
