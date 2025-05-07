from setuptools import setup, find_packages

setup(
    name="crowd_management",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "open3d-python",
    ],
) 