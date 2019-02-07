# -*- coding: utf-8 -*-
#
# Some basic setup for now
#
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = "tea",
    version = "0.0.1",
    author = "Tao Hu",
    author_email = "taohu88@gmail.com",
    description = ("Tea, a deep learning app framework, enables building deep learning apps fast"),
    license = license,
    keywords = "deep learning app framework",
    url = "https://github.com/taohu88/tea",
    packages=find_packages(),
    long_description=readme,
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        "Topic :: App framework",
        "License :: OSI Approved :: MIT License",
    ],
)