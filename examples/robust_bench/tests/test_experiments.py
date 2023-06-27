from typing import Dict

import pytest
from hydra.core.global_hydra import GlobalHydra

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

module = "mart"


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


@RunIf(sh=True)
def test_cifar10_cnn_autoattack_experiment(classification_cfg, tmp_path):
    """Test CIFAR10 CNN AutoAttack experiment."""
    overrides = classification_cfg["datamodel"]
    command = [
        "-m",
        module,
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
