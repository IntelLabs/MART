# Run Attack in Armory

This tutorial shows how to run a MART defined attack in a predefined Armory scenario [carla_obj_det_adversarialpatch_undefended.json](https://github.com/twosixlabs/armory/blob/master/scenario_configs/eval6/carla_overhead_object_detection/carla_obj_det_adversarialpatch_undefended.json).

1. Install Armory and its dependency.

```sh
pip install adversarial-robustness-toolbox==1.12.1
pip install tensorflow==2.10.0
pip install tensorflow-datasets==4.6.0
pip install boto3==1.25.5
pip install ffmpeg-python==0.2.0

pip install armory-testbed==0.16.0
```

2. Copy the YAML config file from a MART experiment result which has defined an attack in test phase in `cfg.model.modules.input_test_adv`. We only need two fields `cfg.model.modules` and `cfg.model.test_sequence` for running the attack in Armory. We should delete any interpolation tokens (like `${path}`) in the config too.

<details><summary>Click to see an example of mart_exp_config.yaml</summary>

<p>

```yaml
model:
  _target_: mart.models.LitModular
  modules:
    input_adv_training:
      _target_: mart.attack.adversary.Dummy
    input_adv_validation:
      _target_: mart.attack.adversary.Dummy
    input_adv_test:
      _target_: mart.attack.adversary.Adversary
      threat_model:
        _target_: mart.attack.BatchThreatModel
        threat_model:
          _target_: mart.attack.threat_model.Overlay
      perturber:
        _target_: mart.attack.BatchPerturber
        perturber_factory:
          _target_: mart.attack.Perturber
          _partial_: true
      generator:
        _target_: mart.attack.adversary.IterativeGenerator
        _partial_: true
        optimizer:
          _target_: torch.optim.SGD
          _partial_: true
          lr: 5
          momentum: 0
          maximize: true
        initializer:
          _target_: mart.attack.initializer.Constant
          constant: 127
        gradient_modifier:
          _target_: mart.attack.gradient_modifier.Sign
        projector:
          _target_: mart.attack.projector.Compose
          projectors:
          - _target_: mart.attack.projector.Mask
          - _target_: mart.attack.projector.Range
            quantize: false
            min: 0
            max: 255
        max_iters: 50
        callbacks:
          progress_bar:
            _target_: mart.attack.callbacks.ProgressBar
          image_visualizer:
            _target_: mart.attack.callbacks.PerturbedImageVisualizer
            folder: adversarial_examples
      objective:
        _target_: mart.nn.CallWith
        module:
          _target_: mart.attack.objective.ZeroAP
          iou_threshold: 0.5
          confidence_threshold: 0.0
        arg_keys:
        - preds
        - target
        kwarg_keys: null
      gain:
        _target_: mart.nn.CallWith
        module:
          _target_: mart.nn.Sum
        arg_keys:
        - rpn_loss.loss_objectness
        - rpn_loss.loss_rpn_box_reg
        - box_loss.loss_classifier
        - box_loss.loss_box_reg
        kwarg_keys: null
    preprocessor:
      _target_: mart.transforms.TupleTransforms
      transforms:
        _target_: torchvision.transforms.Normalize
        mean: 0
        std: 255
    losses_and_detections:
      _target_: mart.models.DualModeGeneralizedRCNN
      model:
        _target_: mart.nn.load_state_dict
        weights_fpath: null
        model:
          _target_: torchvision.models.detection.fasterrcnn_resnet50_fpn
          num_classes: 3
          weights: null
    loss:
      _target_: mart.nn.Sum
    output:
      _target_: mart.nn.ReturnKwargs

  test_sequence:
  - input_adv_test:
      _call_with_args_:
      - input
      - target
      model: model
      step: step
  - preprocessor:
    - input_adv_test
  - losses_and_detections:
    - preprocessor
    - target
  - output:
      preds: losses_and_detections.eval
      target: target
      rpn_loss.loss_objectness: losses_and_detections.training.loss_objectness
      rpn_loss.loss_rpn_box_reg: losses_and_detections.training.loss_rpn_box_reg
      box_loss.loss_classifier: losses_and_detections.training.loss_classifier
      box_loss.loss_box_reg: losses_and_detections.training.loss_box_reg

```

</p>
</details>

3. Change the attack configuration in the Armory scenario.

```diff
    "attack": {                                                     "attack": {
        "knowledge": "white",                                           "knowledge": "white",
        "kwargs": {                                                     "kwargs": {
            "batch_size": 1,                                  |             "mart_exp_config_yaml": "mart_exp_config.yaml"
            "learning_rate": 0.003,                           <
            "max_iter": 1000,                                 <
            "optimizer": "pgd",                               <
            "targeted": false,                                <
            "verbose": true                                   <
        },                                                              },
        "module": "armory.art_experimental.attacks.carla_obj_ |         "module": "mart.attack.adversary_in_art",
        "name": "CARLAAdversarialPatchPyTorch",               |         "name": "MartToArtAttackAdapter",
        "use_label": true                                               "use_label": true
    },                                                              },
```

4. Run the armory scenario.

```console
$ armory run carla_obj_det_adversarialpatch_undefended.json --no-docker --check
...
Evaluation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.05s/it]
2022-11-02 15:53:46 32s SUCCESS  armory.instrument.config:_write:224 benign_carla_od_AP_per_class on benign examples w.r.t. ground truth labels: {'mean': 0.775, 'class': {1: 0.55, 2: 1.0}}
2022-11-02 15:53:46 32s SUCCESS  armory.instrument.config:_write:224 adversarial_carla_od_AP_per_class on adversarial examples w.r.t. ground truth labels: {'mean': 0.395, 'class': {1: 0.55, 2: 0.24}}
...
```
