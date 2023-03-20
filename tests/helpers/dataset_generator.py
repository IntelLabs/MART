#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pathlib

import torch

import tests.helpers.dataset_utils as dataset_utils


class FakeCOCODataset:
    """Class used to generate fake COCO detection datasets.

    This class creates the basic structure that represents the
    COCO dataset:

    coco/
    ├── annotations
    │   ├── instances_test2017.json
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── test
    |   ├── 000000000000.jpg
    |   ⋮
    │   └── 00000000000n.jpg
    ├── train
    |   ├── 000000000000.jpg
    |   ⋮
    │   └── 00000000000n.jpg
    └── val
        ├── 000000000000.jpg
        ⋮
        └── 00000000000n.jpg

    For each group of data you can specify `modalities` to store images.

    Attributes
    ----------
    root : str
        Path to the dataset's root directory.
    config : dict
        Dictionary that describes the dataset structure.
    image_size : Tuple
        Shape of the image data.
    name : str
        Name of the dataset to create.
    """

    def __init__(self, root, config, image_size=(3, 32, 32), name="coco"):
        assert "train" in config
        assert "val" in config
        assert "test" in config

        self.root = root / name
        self.image_size = image_size
        self.config = config
        self.ann_folder = "annotations"
        self.ann_file = "coco_annotations.json"

    def _create_annotations(self, ann_path, name, file_names, num_annotations_per_image):
        """Cretes the COCO annotations.

        Args:
            ann_path (str): Path for annotations file.
            name (str): Annotation's filename.
            file_names (list): List of images to add the annotations.
            num_annotations_per_image (int): Number of annotations for each image.
        """
        image_ids = [int(file_name.stem) for file_name in file_names]
        images = [
            dict(file_name=str(file_name), id=id) for file_name, id in zip(file_names, image_ids)
        ]

        # generate fake mask
        _, h, w = self.image_size
        mask = torch.randint(0, 2, (w, h), dtype=torch.float32)

        # construct the annotations
        rle = dataset_utils.mask_to_rle(mask.cpu().detach().numpy())
        annotations = dataset_utils.combinations_grid(
            image_id=image_ids,
            iscrowd=[0],
            category_id=[0],
            bbox=([1.0, 2.0, 3.0, 4.0],) * num_annotations_per_image,
        )

        for id, annotation in enumerate(annotations):
            annotation["id"] = id
            annotation["segmentation"] = rle
            annotation["area"] = 20

        # store the annotation in a JSON file
        dataset_utils.create_json(ann_path, name, dict(images=images, annotations=annotations))

    def generate(self, num_images, num_annotations_per_image):
        """Generates the corresponding fake dataset.

        Args:
            num_images: number of images.
            num_annotations_per_image: number of annotation for each image.
        """
        # create root directory
        pathlib.Path.mkdir(self.root, exist_ok=True)

        # generate fake images and annotations
        for group in self.config:
            group_info = self.config[group]

            # verify group structure
            assert "folder" in group_info and type(group_info["folder"]) is str
            assert "modalities" in group_info and type(group_info["modalities"]) is list
            assert "ann_folder" in group_info and type(group_info["ann_folder"]) is str
            assert "ann_file" in group_info and type(group_info["ann_file"]) is str

            # find the folders to add fake images
            image_folders = []
            for modality in group_info["modalities"]:
                img_path = pathlib.Path(group_info["folder"]) / modality
                image_folders.append(img_path)

            if len(image_folders) == 0:
                img_path = pathlib.Path(group_info["folder"])
                image_folders.append(img_path)

            # create the fake images
            file_names = []
            for folder in image_folders:
                files = dataset_utils.create_image_folder(
                    self.root,
                    name=folder,
                    file_name_fn=lambda idx: f"{idx:012d}.jpg",
                    num_examples=num_images,
                    size=self.image_size,
                )
                file_names = [file.relative_to(pathlib.Path(self.root) / folder) for file in files]

            # generate the fake annotations
            annotations_path = pathlib.Path(self.root) / group_info["ann_folder"]
            pathlib.Path.mkdir(annotations_path, exist_ok=True)
            self._create_annotations(
                annotations_path,
                group_info["ann_file"],
                file_names,
                num_annotations_per_image,
            )
