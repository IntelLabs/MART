# How to use Modular Adversarial Robustness Toolkit

We provide examples on how to use the toolkit in your project.

A typical procedure is

1. Install the toolkit as a Python package `pip install https://github.com/IntelLabs/MART/archive/refs/heads/main.zip`;
2. Create a `configs` folder;
3. Add your configurations in `configs`;
4. Run experiments at the folder that contains `configs`.

The toolkit searches configurations in the order of `./configs` and `mart.configs`.
Local configurations in `./configs` precede those built-in configurations in `mart/configs`.

You can find specific examples in sub-folders.
