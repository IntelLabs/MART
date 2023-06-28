import os
from typing import Dict

import pytest
from hydra.core.global_hydra import GlobalHydra

from tests.helpers.dataset_generator import FakeCOCODataset
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

module = "mart"

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
def classification_cfg() -> Dict:
    cfg = {
        "trainer": [
            "++trainer.fast_dev_run=3",
        ],
        "datamodel": [
            "datamodule=dummy_classification",
            "datamodule.ims_per_batch=2",
            "datamodule.num_workers=0",
        ],
    }
    yield cfg

    GlobalHydra.instance().clear()


# common configuration for detection related tests.
@pytest.fixture(scope="function")
def coco_cfg(tmp_path) -> Dict:
    # Generate fake COCO dataset on disk at tmp_path
    dataset = FakeCOCODataset(tmp_path, config=coco_ds)
    dataset.generate(num_images=2, num_annotations_per_image=2)

    cfg = {
        "trainer": [
            "++trainer.fast_dev_run=1",
        ],
        "datamodel": [
            "++paths.data_dir=" + str(tmp_path),
            "datamodule.num_workers=0",
        ],
    }
    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def carla_cfg(tmp_path) -> Dict:
    # Generate fake CARLA dataset on disk at tmp_path
    dataset = FakeCOCODataset(tmp_path, config=carla_ds, name="carla_over_obj_det")
    dataset.generate(num_images=2, num_annotations_per_image=2)

    cfg = {
        "trainer": [
            "++trainer.fast_dev_run=3",
        ],
        "datamodel": [
            "++paths.data_dir=" + str(tmp_path),
            "datamodule.num_workers=0",
        ],
    }
    yield cfg

    GlobalHydra.instance().clear()


@RunIf(sh=True)
def test_cifar10_cnn_adv_experiment(classification_cfg, tmp_path):
    """Test CIFAR10 CNN experiment."""
    overrides = classification_cfg["trainer"] + classification_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=CIFAR10_CNN_Adv",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.modules.input_adv_test.max_iters=10",
        "optimized_metric=training_metrics/acc",
        "++datamodule.train_dataset.image_size=[3,32,32]",
        "++datamodule.train_dataset.num_classes=10",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
def test_cifar10_cnn_experiment(classification_cfg, tmp_path):
    """Test CIFAR10 CNN experiment."""
    overrides = classification_cfg["trainer"] + classification_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=CIFAR10_CNN",
        "hydra.sweep.dir=" + str(tmp_path),
        "optimized_metric=training_metrics/acc",
        "++datamodule.train_dataset.image_size=[3,32,32]",
        "++datamodule.train_dataset.num_classes=10",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
def test_cifar10_cnn_autoattack_experiment(classification_cfg, tmp_path):
    """Test CIFAR10 CNN AutoAttack experiment."""
    overrides = classification_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=CIFAR10_CNN",
        "hydra.sweep.dir=" + str(tmp_path),
        "++datamodule.train_dataset.image_size=[3,32,32]",
        "++datamodule.train_dataset.num_classes=10",
        "fit=false",
        "+attack@model.modules.input_adv_test=classification_autoattack",
        '+model.modules.input_adv_test.adversary.partial.device="cpu"',
        "+trainer.limit_test_batches=1",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_imagenet_timm_experiment(classification_cfg, tmp_path):
    """Test ImageNet Timm experiment."""
    overrides = classification_cfg["trainer"] + classification_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=ImageNet_Timm",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.precision=32",
        "optimized_metric=training_metrics/acc",
        "++datamodule.train_dataset.image_size=[3,469,387]",
        "++datamodule.train_dataset.num_classes=200",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_coco_fasterrcnn_experiment(coco_cfg, tmp_path):
    """Test TorchVision FasterRCNN experiment."""
    overrides = coco_cfg["trainer"] + coco_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=COCO_TorchvisionFasterRCNN",
        "hydra.sweep.dir=" + str(tmp_path),
        "optimized_metric=training/loss_objectness",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_coco_fasterrcnn_adv_experiment(coco_cfg, tmp_path):
    """Test TorchVision FasterRCNN Adv experiment."""
    overrides = coco_cfg["trainer"] + coco_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=COCO_TorchvisionFasterRCNN_Adv",
        "hydra.sweep.dir=" + str(tmp_path),
        "optimized_metric=training/loss_objectness",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_coco_retinanet_experiment(coco_cfg, tmp_path):
    """Test TorchVision RetinaNet experiment."""
    overrides = coco_cfg["trainer"] + coco_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=COCO_TorchvisionRetinaNet",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer.precision=32",
        "optimized_metric=training/loss_box_reg",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_armory_carla_fasterrcnn_experiment(carla_cfg, tmp_path):
    """Test Armory CARLA TorchVision FasterRCNN experiment."""
    overrides = carla_cfg["trainer"] + carla_cfg["datamodel"]
    command = [
        module,
        "-m",
        "experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN",
        "+attack@model.modules.input_adv_test=object_detection_mask_adversary",
        "hydra.sweep.dir=" + str(tmp_path),
        "optimized_metric=training/loss_objectness",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_resume(tmpdir):
    # Create a pseudo folder to resume from.
    ckpt = tmpdir.mkdir("checkpoints").join("last.ckpt")
    yaml_dir = tmpdir.mkdir(".hydra")
    config_yaml = yaml_dir.join("config.yaml")
    hydra_yaml = yaml_dir.join("hydra.yaml")
    overrides_yaml = yaml_dir.join("overrides.yaml")

    ckpt.write("")
    config_yaml.write("")
    # hdra.job.config_name is required to compose an experiment.
    hydra_yaml.write("hydra:\n  job:\n    config_name: lightning.yaml")
    # The experiment is usually specified in the original configuration in overrides,
    #   so we should not require experiment=? when resume=? is used.
    # We also replace with a dummy dataset to avoid downloading CIFAR-10.
    overrides_yaml.write(
        "\n".join(
            [
                "- experiment=CIFAR10_CNN",
                "- datamodule=dummy_classification",
                "- datamodule.ims_per_batch=2",
                "- datamodule.num_workers=0",
                "- datamodule.train_dataset.size=2",
                "- datamodule.train_dataset.image_size=[3,32,32]",
                "- datamodule.train_dataset.num_classes=10",
                "- fit=false",  # Don't train or test the model, because the checkpoint is invalid.
                "- test=false",
                "- optimized_metric=null",  # No metric to retrieve.
                "- extras.print_config=false",  # Test if print_config is turned off after resume.
            ]
        )
    )

    # Disable timestamp in the output path so we can find it easily.
    output_dir = tmpdir.mkdir("output")

    command = [module, "resume=" + str(ckpt), f"hydra.run.dir={output_dir}"]
    run_sh_command(command)

    # Test if the job is executed.
    assert os.path.isfile(os.path.join(output_dir, "__main__.log"))

    # Test if print_config is turned off after resume.
    assert not os.path.isfile(os.path.join(output_dir, "config_tree.log"))
