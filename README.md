<div align="center">

# Modular Adversarial Robustness Toolkit

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<img src="data/loop.png" width="600">

<b>A unified optimization-based framework</b>

</div>

## Description

**Modular Adversarial Robustness Toolkit** makes it easy to compose novel attacks to evaluate adversarial robustness of deep learning models. Thanks to the modular design of the optimization-based attack framework, you can use off-the-shelf elements, such as optimizers and learning rate schedulers, from PyTorch to compose powerful attacks. The unified framework also supports advanced features, such as early stopping, to improve attack efficiency.

<div align="center">
  <img src="data/arch.png" width="600">

<b>Modular Design</b>

</div>

## Installation

```bash
# clone project
git clone https://github.com/IntelLabs/MART
cd MART

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# [OPTIONAL] or create virtualenv environment
python3.9 -m venv .venv
source .venv/bin/activate

# install requirements
# you can ignore the [false alarm](https://github.com/tensorflow/tensorboard/pull/5922):
#   ERROR: tensorboard 2.10.1 has requirement protobuf<3.20,>=3.9.2, but you'll have protobuf 3.20.1 which is incompatible.
# we haven't encountered any issue with this progress bar dependency error either:
#   ERROR: pytorch-lightning 1.6.5 has requirement tqdm>=4.57.0, but you'll have tqdm 4.56.2 which is incompatible.
pip install -r requirements.txt

# If your CUDA version is not 10.2, you need to uninstall pytorch and torchvision from requirements.txt,
# then reinstall them according to platform instructions at https://pytorch.org/get-started/
# FYI, this is what we do:
#   $ pip uninstall torch torchvision
#   $ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# [OPTIONAL] install pre-commit hooks
# this will trigger the pre-commit checks in each `git commit` command.
pre-commit install


# [OPTIONAL] install Modular Adversarial Robustness Toolkit, if you plan to create your own `configs` folder elsewhere.
pip install -e .
```

## How to run

The [configs](/configs) folder is required to run the toolkit. You can evaluate adversarial robustness of pretrained models with chosen experiment configuration from [configs/experiment/](configs/experiment/). Feel free to override any parameter from command line. Run `make` to learn more tasks pre-defined in [Makefile](Makefile).

```bash
# run on CPU
python -m mart experiment=CIFAR10_RobustBench \
	trainer=default \
	fit=false \
	+trainer.limit_test_batches=1 \
	+attack@model.modules.input_adv_test=classification_eps8_pgd10_step1

# run on GPU
python -m mart experiment=CIFAR10_RobustBench \
	trainer=gpu \
	fit=false \
	+trainer.limit_test_batches=1 \
	+attack@model.modules.input_adv_test=classification_eps8_pgd10_step1
```

You can also install the repository as a package, then run `python -m mart` from anywhere with your own `configs` folder.

## Acknowledgements

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001119S0026.

## Disclaimer

This “research quality code”  is provided by Intel “As Is” without any express or implied warranty of any kind. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository will not be actively maintained and as such may contain components that are out of date, or contain known security vulnerabilities. Proceed with caution.
