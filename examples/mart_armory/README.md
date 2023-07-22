## Installation

Install the `mart_armory` package from a repo subdirectory.

```shell
pip install 'git+https://github.com/IntelLabs/MART.git@example_armory_attack#egg=mart_armory&subdirectory=examples/mart_armory'
```

## Usage

1. Generate a YAML configuration of attack.

```shell
python -m mart_armory.generate_attack_config \
attack=[object_detection_mask_adversary,data_coco] \
output=path_to_attack.yaml
```

2. Update the attack section in the Armory scenario configuration.

```json
"attack": {
    "kwargs": {
        "mart_adv_config_yaml": "path/to/attack.yaml",
    },
    "module": "mart_armory",
    "name": "MartAttackObjectDetection",
},
```
