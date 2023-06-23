# https://raw.githubusercontent.com/pytorch/vision/ae30df455405fb56946425bf3f3c318280b0a7ae/torchvision/models/detection/anchor_utils.py
import torch
from torch import Tensor


def grid_offsets(grid_size: Tensor) -> Tensor:
    """Given a grid size, returns a tensor containing offsets to the grid cells.

    Args:
        The width and height of the grid in a tensor.

    Returns:
        A ``[height, width, 2]`` tensor containing the grid cell `(x, y)` offsets.
    """
    x_range = torch.arange(grid_size[0].item(), device=grid_size.device)
    y_range = torch.arange(grid_size[1].item(), device=grid_size.device)
    grid_y, grid_x = torch.meshgrid([y_range, x_range], indexing="ij")
    return torch.stack((grid_x, grid_y), -1)


def grid_centers(grid_size: Tensor) -> Tensor:
    """Given a grid size, returns a tensor containing coordinates to the centers of the grid cells.

    Returns:
        A ``[height, width, 2]`` tensor containing coordinates to the centers of the grid cells.
    """
    return grid_offsets(grid_size) + 0.5


@torch.jit.script
def global_xy(xy: Tensor, image_size: Tensor) -> Tensor:
    """Adds offsets to the predicted box center coordinates to obtain global coordinates to the image.

    The predicted coordinates are interpreted as coordinates inside a grid cell whose width and height is 1. Adding
    offset to the cell, dividing by the grid size, and multiplying by the image size, we get global coordinates in the
    image scale.

    The function needs the ``@torch.jit.script`` decorator in order for ONNX generation to work. The tracing based
    generator will loose track of e.g. ``xy.shape[1]`` and treat it as a Python variable and not a tensor. This will
    cause the dimension to be treated as a constant in the model, which prevents dynamic input sizes.

    Args:
        xy: The predicted center coordinates before scaling. Values from zero to one in a tensor sized
            ``[batch_size, height, width, boxes_per_cell, 2]``.
        image_size: Width and height in a vector that will be used to scale the coordinates.

    Returns:
        Global coordinates scaled to the size of the network input image, in a tensor with the same shape as the input
        tensor.
    """
    height = xy.shape[1]
    width = xy.shape[2]
    grid_size = torch.tensor([width, height], device=xy.device)
    # Scripting requires explicit conversion to a floating point type.
    offset = grid_offsets(grid_size).to(xy.dtype).unsqueeze(2)  # [height, width, 1, 2]
    scale = torch.true_divide(image_size, grid_size)
    return (xy + offset) * scale
