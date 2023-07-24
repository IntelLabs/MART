## Installation

Install the `mart_armory` package from a repo subdirectory.

```shell
pip install 'git+https://github.com/IntelLabs/MART.git@example_armory_attack#egg=mart_armory&subdirectory=examples/mart_armory'
```

## Usage

1. Generate a YAML configuration of attack.

```shell
python -m mart_armory.generate_attack_config \
batch_converter=object_detection \
model_wrapper=art_rcnn \
attack=[object_detection_mask_adversary,data_coco] \
attack.objective=null \
output=path/to/attack.yaml
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

Armory requires the argument `knowledge`. The statement `"use_label": true` gets `y` for the attack.
