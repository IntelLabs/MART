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

### Using pip

```bash
pip install mart[full]@https://github.com/IntelLabs/MART/archive/refs/tags/<VERSION>.zip
```

Replace `<VERSION>` with the MART's version you want to install. For example:

```bash
pip install mart[full]@https://github.com/IntelLabs/MART/archive/refs/tags/v0.2.1.zip
```

### Manual installation

```bash
# clone project
git clone https://github.com/IntelLabs/MART
cd MART

# [OPTIONAL] create conda environment
# Recommend Python 3.9
conda create -n myenv python=3.9
conda activate myenv

# [OPTIONAL] or create virtualenv environment
python3 -m venv .venv
source .venv/bin/activate

# Install Modular Adversarial Robustness Toolkit, if you plan to create your own `configs` folder elsewhere.
pip install -e .[full]

# [OPTIONAL] install pre-commit hooks
# this will trigger the pre-commit checks in each `git commit` command.
pre-commit install

# If your CUDA version is not 10.2, you need to uninstall pytorch and torchvision, and
# then reinstall them according to platform instructions at https://pytorch.org/get-started/
# FYI, this is what we do:
#   $ pip uninstall torch torchvision
#   $ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

```

## How to run

The toolkit comes with built-in experiment configurations in [mart/configs](mart/configs).

For example, you can run a fast adversarial training experiment on CIFAR-10 with `python -m mart experiment=CIFAR10_CNN_Adv`.
Running on GPU will make it even faster `CUDA_VISIBLE_DEVICES=0 python -m mart experiment=CIFAR10_CNN_Adv trainer=gpu trainer.precision=16`.

You can see other examples in [examples](/examples).

## Acknowledgements

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001119S0026.

## Disclaimer

This “research quality code” is provided by Intel “As Is” without any express or implied warranty of any kind. Intel does not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security review has not been performed on this code. Additionally, this repository will not be actively maintained and as such may contain components that are out of date, or contain known security vulnerabilities. Proceed with caution.
