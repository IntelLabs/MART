from typing import Dict

import pytest
from hydra.core.global_hydra import GlobalHydra

from tests.helpers.dataset_generator import FakeCOCODataset
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

MODULE = "mart"


def run_mart_command(*args):
    return run_sh_command(
        ["-m", MODULE, "++trainer.fast_dev_run=3", "hydra.sweep.dir=/tmp", *args]
    )


coco_ds = {
    "train": {
        "folder": "train2017",
        "modalities": [],
        "ann_folder": "annotations",
        "ann_file": "instances_train2017.json",
    },
    "val": {
        "folder": "val2017",
        "modalities": [],
        "ann_folder": "annotations",
        "ann_file": "instances_val2017.json",
    },
    "test": {
        "folder": "test2017",
        "modalities": [],
        "ann_folder": "annotations",
        "ann_file": "instances_test2017.json",
    },
}

carla_ds = {
    "train": {
        "folder": "train",
        "modalities": ["rgb"],
        "ann_folder": "train",
        "ann_file": "kwcoco_annotations.json",
    },
    "val": {
        "folder": "val",
        "modalities": ["rgb"],
        "ann_folder": "val",
        "ann_file": "kwcoco_annotations.json",
    },
    "test": {
        "folder": "dev",
        "modalities": ["foreground_mask", "rgb"],
        "ann_folder": "dev",
        "ann_file": "kwcoco_annotations.json",
    },
}


# common configuration for classification related tests.
@pytest.fixture(scope="function")
def classification_overrides(tmp_path) -> Dict:
    yield [
        "datamodule=dummy_classification",
        "datamodule.ims_per_batch=2",
        "datamodule.num_workers=0",
        "++datamodule.train_dataset.image_size=[3,32,32]",
        "++datamodule.train_dataset.num_classes=10",
    ]

    GlobalHydra.instance().clear()


# common configuration for detection related tests.
@pytest.fixture(scope="function")
def coco_overrides(tmp_path) -> Dict:
    # Generate fake COCO dataset on disk at tmp_path
    dataset = FakeCOCODataset(tmp_path, config=coco_ds)
    dataset.generate(num_images=2, num_annotations_per_image=2)

    yield [
        "++paths.data_dir=" + str(tmp_path),
        "datamodule.num_workers=0",
    ]

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def carla_overrides(tmp_path) -> Dict:
    # Generate fake CARLA dataset on disk at tmp_path
    dataset = FakeCOCODataset(tmp_path, config=carla_ds, name="carla_over_obj_det")
    dataset.generate(num_images=2, num_annotations_per_image=2)

    yield [
        "++paths.data_dir=" + str(tmp_path),
        "datamodule.num_workers=0",
    ]

    GlobalHydra.instance().clear()


@RunIf(sh=True)
def test_cifar10_cnn_adv_experiment(classification_overrides):
    """Test CIFAR10 CNN experiment."""
    run_mart_command(
        "experiment=CIFAR10_CNN_Adv",
        "model.modules.input_adv_test.max_iters=3",
        *classification_overrides
    )


@RunIf(sh=True)
def test_cifar10_cnn_experiment(classification_overrides):
    """Test CIFAR10 CNN experiment."""
    run_mart_command("experiment=CIFAR10_CNN", *classification_overrides)


@RunIf(sh=True)
def test_cifar10_robust_bench_experiment(classification_overrides):
    """Test CIFAR10 Robust Bench experiment."""
    run_mart_command("experiment=CIFAR10_RobustBench", *classification_overrides)


@RunIf(sh=True)
@pytest.mark.slow
def test_imagenet_timm_experiment(classification_overrides):
    """Test ImageNet Timm experiment."""
    run_mart_command(
        "experiment=ImageNet_Timm",
        "++datamodule.train_dataset.image_size=[3,469,387]",
        "++datamodule.train_dataset.num_classes=200",
        "trainer.precision=32",  # CPU only supports 32-bit
        *classification_overrides
    )


@RunIf(sh=True)
@pytest.mark.slow
def test_coco_fasterrcnn_experiment(coco_overrides):
    """Test TorchVision FasterRCNN experiment."""
    run_mart_command("experiment=COCO_TorchvisionFasterRCNN", *coco_overrides)


@RunIf(sh=True)
@pytest.mark.slow
def test_coco_fasterrcnn_adv_experiment(coco_overrides):
    """Test TorchVision FasterRCNN Adv experiment."""
    run_mart_command(
        "experiment=COCO_TorchvisionFasterRCNN_Adv",
        "model.modules.input_adv_test.max_iters=3",
        *coco_overrides
    )


@RunIf(sh=True)
@pytest.mark.slow
def test_coco_retinanet_experiment(coco_overrides):
    """Test TorchVision RetinaNet experiment."""
    run_mart_command(
        "experiment=COCO_TorchvisionRetinaNet",
        "trainer.precision=32",  # CPU only supports 32-bit
        *coco_overrides
    )


@RunIf(sh=True)
@pytest.mark.slow
def test_armory_carla_fasterrcnn_experiment(carla_overrides):
    """Test Armory CARLA TorchVision FasterRCNN experiment."""
    run_mart_command("experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN", *carla_overrides)
