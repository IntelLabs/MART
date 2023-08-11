# Introduction

This example shows how to use MART with the FiftyOne integration. FiftyOne is an open-source tool for building high quality dataset of images and videos. With this integration, MART delegates the data handling to FiftyOne.

## Installation

```bash
pip install git+https://github.com/IntelLabs/MART.git[fiftyone]
```

# FiftyOne commands to load (index) datasets.

Use COCO-2017 as an example. Unfortunately, FiftyOne does not support person-keypoints annotations yet.

## Download and load zoo datasets

```bash
fiftyone zoo datasets load \
coco-2017 \
-s train \
-n coco-2017-instances-train \
-k include_id=true label_types=detections,segmentations

fiftyone zoo datasets load \
coco-2017 \
-s validation \
-n coco-2017-instances-validation \
-k include_id=true label_types=detections,segmentations
```

## Load local datasets

```bash
fiftyone datasets create \
--name coco-2017-instances-validation \
--dataset-dir /path/to/datasets/coco/ \
--type fiftyone.types.COCODetectionDataset \
--kwargs \
data_path="val2017" \
labels_path=/path/to/datasets/coco/annotations/instances_val2017.json \
persistent=true \
include_id=true
```

## Use the FiftyOne datamodule

```yaml
datamodule:
  train_dataset:
    dataset_name: coco-2017-instances-train
    gt_field: segmentations
  val_dataset:
    dataset_name: coco-2017-instances-validation
    gt_field: segmentations
```
