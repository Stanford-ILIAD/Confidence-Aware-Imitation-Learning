#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="cail",
    version="0.0.0",
    description='Official implementation of the NeurIPS 2021 paper: S Zhang, Z Cao, D Sadigh, Y Sui: '
                '"Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality"',
    author="Songyuan Zhang",
    author_email="szhang21@mit.edu",
    url="https://github.com/syzhang092218-source/Confidence-Aware-Imitation-Learning",
    install_requires=[],
    packages=find_packages(),
)
