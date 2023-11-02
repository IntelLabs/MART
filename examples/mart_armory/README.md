## Introduction

We demonstrate how to configure and run a MART attack against object detection models in ARMORY.

The demo attack here is about 30% faster than the baseline attack implementation in ARMORY, because we specify to use 16bit-mixed-precision in the MART attack configuration.

MART is designed to be modular and configurable. It should empower users to evaluate adversarial robustness of deep learning models more effectively and efficiently.

Please reach out to [Weilin Xu](mailto:weilin.xu@intel.com) if you have any question.

## Installation

Download the code repositories.

```shell
# You can start from any directory other than `~/coder/`, since we always use relative paths in the following commands.
mkdir ~/coder; cd ~/coder

git clone https://github.com/twosixlabs/armory.git
# Make sure we are on the same page.
cd armory; git checkout tags/v0.19.0 -b r0.19.0; cd ..

git clone https://github.com/IntelLabs/MART.git -b example_armory_attack
```

Create and activate a Python virtualen environment.

```shell
cd armory
python -m venv .venv
source .venv/bin/activate
```

Install ARMORY, MART and the glue package mart_armory in editable mode.

```shell
pip install -e .[engine]
pip install tensorflow tensorflow-datasets
# PyTorch 2.0+ is already in the dependency of MART.
pip install -e ../MART
pip install -e ../MART/examples/mart_armory
```

Make sure PyTorch works on CUDA.

```console
$ CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.is_available())"
True
```

> You may need to install a different PyTorch distribution if your CUDA is not 12.0.

> Here's my `nvidia-smi` output: `| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |`

## Usage

1. Generate a YAML configuration of attack, using Adam as the optimizer.

```shell
python -m mart.generate_config \
--config_dir=../MART/examples/mart_armory/mart_armory/configs \
--config_name=assemble_attack.yaml \
batch_converter=object_detection \
batch_c15n=data_coco \
attack=[object_detection_mask_adversary] \
+attack.precision=16 \
attack.optimizer.optimizer.path=torch.optim.Adam \
~attack.optimizer.momentum \
attack.objective=null \
attack.max_iters=500 \
attack.lr=13 \
model_transform=armory_objdet \
> mart_objdet_attack_adam500.yaml
```

2. Run the MART attack on one example.

```shell
cat scenario_configs/eval7/carla_overhead_object_detection/carla_obj_det_adversarialpatch_undefended.json \
| jq 'del(.attack)' \
| jq '.attack.knowledge="white"' \
| jq '.attack.use_label=true' \
| jq '.attack.module="mart_armory"' \
| jq '.attack.name="MartAttack"' \
| jq '.attack.kwargs.mart_adv_config_yaml="mart_objdet_attack_adam500.yaml"' \
| jq '.scenario.export_batches=true' \
| CUDA_VISIBLE_DEVICES=0 armory run - --no-docker --use-gpu --gpus=1 --num-eval-batches 1
```

```
2023-10-13 12:05:33 1m14s METRIC   armory.instrument.instrument:_write:743 adversarial_object_detection_mAP_tide on adversarial examples w.r.t. ground truth labels: {'mAP': {0.5: 0.0, 0.55: 0.0, 0.6: 0.0, 0.65: 0.0, 0.7: 0.0, 0.75: 0.0, 0.8: 0.0, 0.85: 0.0, 0.9: 0.0, 0.95: 0.0}, 'errors': {'main': {'dAP': {'Cls': 0.0, 'Loc': 0.0, 'Both': 0.0, 'Dupe': 0.0, 'Bkg': 0.0, 'Miss': 0.0}, 'count': {'Cls': 0, 'Loc': 0, 'Both': 0, 'Dupe': 0, 'Bkg': 100, 'Miss': 21}}, 'special': {'dAP': {'FalsePos': 0.0, 'FalseNeg': 0.0}, 'count': {'FalseNeg': 21}}}}
```

## Comparison

Run the baseline attack on the same example for comparison. The MART attack is ~30% faster due to the 16-bit mixed precision.

```shell
cat scenario_configs/eval7/carla_overhead_object_detection/carla_obj_det_adversarialpatch_undefended.json \
| jq '.scenario.export_batches=true' \
| CUDA_VISIBLE_DEVICES=0 armory run - --no-docker --use-gpu --gpus=1 --num-eval-batches 1
```

```console
2023-10-13 12:11:50 1m33s METRIC   armory.instrument.instrument:_write:743 adversarial_object_detection_mAP_tide on adversarial examples w.r.t. ground truth labels: {'mAP': {0.5: 0.0, 0.55: 0.0, 0.6: 0.0, 0.65: 0.0, 0.7: 0.0, 0.75: 0.0, 0.8: 0.0, 0.85: 0.0, 0.9: 0.0, 0.95: 0.0}, 'errors': {'main': {'dAP': {'Cls': 0.0, 'Loc': 0.0, 'Both': 0.0, 'Dupe': 0.0, 'Bkg': 0.0, 'Miss': 0.0}, 'count': {'Cls': 0, 'Loc': 0, 'Both': 0, 'Dupe': 0, 'Bkg': 100, 'Miss': 21}}, 'special': {'dAP': {'FalsePos': 0.0, 'FalseNeg': 0.0}, 'count': {'FalseNeg': 21}}}}
```
