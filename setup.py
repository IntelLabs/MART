#!/usr/bin/env python

import codecs
import os.path

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="mart",
    version=get_version("mart/__init__.py"),
    description="Modular Adversarial Robustness Toolkit",
    author="Intel Corporation",
    author_email="weilin.xu@intel.com",
    url="https://github.com/IntelLabs/MART",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
