from setuptools import setup, find_packages

setup(
    name="autonnx",
    packages=find_packages(),
    install_requires=[
        "rich",
        "torch",
    ],
    version="0.1",
)
