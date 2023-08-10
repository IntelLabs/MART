# Introduction

This example shows how to use MART with the FiftyOne integration. FiftyOne is an open-source tool for building high quality dataset of images and videos. With this integration, MART delegates the data handling to FiftyOne.

## Installation

```bash
pip install git+https://github.com/IntelLabs/MART.git[fiftyone]
```

## How to run

### Using the Dataset ZOO

- First, identify the dataset available in the [FiftyOne Dataset Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html). Take into account the the current implementation support COCO like datasets.

- Run the following command to use the ZOO dataset in MART:

```bash
python -m mart experiment=COCO_TorchvisionFasterRCNN \
    trainer.max_steps=5105 \
    datamodule=fiftyone \
    datamodule.train_dataset.dataset_name="coco-2017" \
    datamodule.train_dataset.gt_field="ground_truth" \
    +datamodule.train_dataset.label_types=["segmentations"] \
    +datamodule.train_dataset.classes=["person","car","motorcycle"] \
    +datamodule.train_dataset.split="train" \
    +datamodule.train_dataset.max_samples=1000 \
    datamodule.val_dataset.dataset_name="coco-2017" \
    +datamodule.val_dataset.label_types=["segmentations"] \
    +datamodule.val_dataset.classes=["person","car","motorcycle"] \
    +datamodule.val_dataset.split="validation" \
    +datamodule.val_dataset.max_samples=50 \
    datamodule.test_dataset.dataset_name="coco-2017" \
    +datamodule.test_dataset.label_types=["segmentations"] \
    +datamodule.test_dataset.classes=["person","car","motorcycle"] \
    +datamodule.test_dataset.split="validation" \
    +datamodule.test_dataset.max_samples=50 \
    +datamodule.test_dataset.shuffle=true
```

To configure the dataset, consult the corresponding documentation to know the available config options (E.g. [coco-2017](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#dataset-zoo-coco-2017))

### Using a custom dataset

- Add an existing dataset into FiftyOne by running the following command:

```bash
fiftyone datasets create \
    --name <DATASET_NAME> \
    --dataset-dir <PATH/TO/DATASET> \
    --type fiftyone.types.COCODetectionDataset \
    --kwargs \
    data_path="<PATH/TO/DATA>" \
    labels_path=<PATH/TO/ANNOTATIONS> \
    persistent=true \
    include_id=true
```

- Use the custom dataset in MART with this command:

```bash
python -m mart experiment=COCO_TorchvisionFasterRCNN \
    datamodule=fiftyone \
    datamodule.train_dataset.dataset_name="train_dataset" \
    datamodule.val_dataset.dataset_name="test_dataset" \
    datamodule.test_dataset.dataset_name="test_dataset" \
    datamodule.train_dataset.sample_tags=["tag_name"] \
    datamodule.train_dataset.label_tags=["tag_name_1","tag_name_2","tag_name_3"]
```

Notice that in the above example is possible to filter samples and annotations by using the FiftyOne tags.
