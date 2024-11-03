import torch
from torch import Tensor

from utils.conv import torch_conv

SOBEL_X: Tensor = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32)

SOBEL_Y: Tensor = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32)


def sobel(image: Tensor) -> Tensor:
    edge_x = torch_conv(image, SOBEL_X, padding=1)
    edge_y = torch_conv(image, SOBEL_Y, padding=1)

    # compute the gradient magnitude
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge_magnitude

