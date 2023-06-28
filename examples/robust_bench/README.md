# Introduction

This example shows how to use MART to evaluate adversarial robustness of models from RobustBench.

Note that the attack algorithm here is not optimal, just for the demonstration purpose.

The `requirements.txt` contains dependency of MART and RobustBench.

The `./configs` folder contains configurations of the target model `classifier_robustbench` and the MART experiment `CIFAR10_RobustBench`.

The configuration files in `./configs` precedes those in `mart.configs` (MART's built-in configs).

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
# run on CPU
python -m mart experiment=CIFAR10_RobustBench \
	trainer=default \
	fit=false \
	+trainer.limit_test_batches=1 \
	+attack@model.modules.input_adv_test=classification_eps8_pgd10_step1

# run on GPU
CUDA_VISIBLE_DEVICES=0 \
python -m mart experiment=CIFAR10_RobustBench \
	trainer=gpu \
	fit=false \
	+trainer.limit_test_batches=1 \
	+attack@model.modules.input_adv_test=classification_eps8_pgd10_step1 \
	+model.test_sequence.seq005=input_adv_test \
	model.test_sequence.seq010.preprocessor=["input_adv_test"]

# Evaluate with AutoAttack, expect 0.6171875
CUDA_VISIBLE_DEVICES=0 \
python -m mart experiment=CIFAR10_RobustBench \
	trainer=gpu \
	fit=false \
	+trainer.limit_test_batches=1 \
	+attack@model.modules.input_adv_test=classification_autoattack \
	+model.test_sequence.seq005=input_adv_test \
	model.test_sequence.seq010.preprocessor=["input_adv_test"]
```
