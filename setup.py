# -*- coding: utf-8 -*-
"""
Setup module for the group_project_se489 package.

This module contains the setup configuration for the package.

Attributes
----------
__version__ : str
    The version of the package.
REPO_NAME : str
    The name of the GitHub repository.
AUTHOR_USER_NAME : str
    The GitHub username of the author.
SRC_REPO : str
    The source repository name.
AUTHOR_EMAIL : str
    The email address of the author.
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "group_project_se489"
AUTHOR_USER_NAME = "RobinMacwan"
SRC_REPO = "se489_group_project"
AUTHOR_EMAIL = "robin_macwan@yahoo.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for kidney scan",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "se489_group_project"},
    packages=setuptools.find_packages(where="se489_group_project"),
)
