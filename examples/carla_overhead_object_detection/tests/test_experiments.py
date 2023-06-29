import os
from typing import Dict

import pytest
from hydra.core.global_hydra import GlobalHydra

from tests.helpers.dataset_generator import FakeCOCODataset
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

module = "mart"


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


@RunIf(sh=True)
@pytest.mark.slow
def test_armory_carla_fasterrcnn_experiment(carla_cfg, tmp_path):
    """Test Armory CARLA TorchVision FasterRCNN experiment."""
    overrides = carla_cfg["trainer"] + carla_cfg["datamodel"]
    command = [
        "-m",
        module,
        "experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN",
        "+attack@model.modules.input_adv_test=object_detection_mask_adversary",
        "hydra.sweep.dir=" + str(tmp_path),
        "optimized_metric=training/loss_objectness",
    ] + overrides
    run_sh_command(command)
