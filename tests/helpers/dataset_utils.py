#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import itertools
import json
import pathlib

import PIL
import torch


def create_image_file(root, name, size, **kwargs):
    """Create an image file from random data.
    Reference: https://github.com/pytorch/vision/blob/7b8a6db7f450e70a1e0fb07e07b30dda6a7e6e1c/test/datasets_utils.py#L675

    Args:
        root (str): Root for the images that will be created.
        name (str): File name.
        size (Sequence[int]): Size of the image that represents the ``(num_channels, height, width)``.
    Returns:
        pathlib.Path: Path to the created image file.
    """

    image = torch.randint(0, 256, size, dtype=torch.uint8)
    file = pathlib.Path(root) / name

    # torch (num_channels x height x width) -> PIL (width x height x num_channels)
    image = image.permute(2, 1, 0)
    PIL.Image.fromarray(image.numpy()).save(file, **kwargs)
    return file


def create_image_folder(root, name, file_name_fn, num_examples, size, **kwargs):
    """Create a folder of random images.
    Reference: https://github.com/pytorch/vision/blob/7b8a6db7f450e70a1e0fb07e07b30dda6a7e6e1c/test/datasets_utils.py#L711

    Args:
        root (str): Root directory the image folder will be placed in.
        name (str): Name of the image folder.
        file_name_fn (Callable[[int], str]): Should return a file name if called with the file index.
        num_examples (int): Number of images to create.
        size (Sequence[int]): Size of the images.
    Returns:
        List[pathlib.Path]: Paths to all created image files.
    """
    root = pathlib.Path(root) / name
    pathlib.Path.mkdir(root, parents=True, exist_ok=True)

    created_files = [
        create_image_file(
            root, file_name_fn(idx), size=size(idx) if callable(size) else size, **kwargs
        )
        for idx in range(num_examples)
    ]

    return created_files


def mask_to_rle(mask):
    """Convert mask to RLE.

    Args:
        mask (numpy.array): Binary mask.
    Returns:
        dict: RLE representation.
    """
    rle = {"counts": [], "size": list(mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(itertools.groupby(mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def create_json(root, name, content):
    """Creates a JSON file with dataset annotations.

    Args:
        root (str): Directory where the annotations will be stored.
        name (str): Annotation's filename.
        content (dict): Dictionary with the annotation's content.
    """
    file = pathlib.Path(root) / name
    with open(file, "w") as fh:
        json.dump(content, fh)
    return


def combinations_grid(**kwargs):
    """Generates an array of parameters combinations.

    Reference: https://github.com/pytorch/vision/blob/7b8a6db7f450e70a1e0fb07e07b30dda6a7e6e1c/test/datasets_utils.py#L172
    """
    return [dict(zip(kwargs.keys(), values)) for values in itertools.product(*kwargs.values())]
