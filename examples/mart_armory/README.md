## Installation

Install the `mart_armory` package from a repo subdirectory.

```shell
pip install 'git+https://github.com/IntelLabs/MART.git@example_armory_attack#egg=mart_armory&subdirectory=examples/mart_armory'
```

## Usage

1. Generate a YAML configuration of attack.

```shell
python -m mart.generate_config \
--config_dir=mart_armory/configs \
--config_name=assemble_attack.yaml \
batch_converter=object_detection \
batch_c15n=data_coco \
attack=[object_detection_mask_adversary] \
attack.objective=null \
attack.max_iters=10 \
attack.lr=26 \
model_transform=armory_objdet \
> path/to/attack.yaml
```

2. Update the attack section in the Armory scenario configuration.

```json
"attack": {
    "module": "mart_armory",
    "name": "MartAttack",
    "kwargs": {
        "mart_adv_config_yaml": "path/to/attack.yaml"
    },
    "knowledge": "white",
    "use_label": true
},
```

Note that Armory requires the argument `knowledge`. The statement `"use_label": true` gets `y` for the attack.

Alternatively, we can use `jq` to update existing scenario json files, for example

```bash
cat scenario_configs/eval7/carla_overhead_object_detection/carla_obj_det_adversarialpatch_undefended.json \
| jq 'del(.attack)' \
| jq '.attack.knowledge="white"' \
| jq '.attack.use_label=true' \
| jq '.attack.module="mart_armory"' \
| jq '.attack.name="MartAttack"' \
| jq '.attack.kwargs.mart_adv_config_yaml="path/to/attack.yaml"' \
| jq '.scenario.export_batches=true' \
| CUDA_VISIBLE_DEVICES=0 armory run - --no-docker --use-gpu --gpus=1
```
