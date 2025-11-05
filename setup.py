from setuptools import setup, find_packages

setup(
    name="baoflamingo",
    version="0.1.0", # just a random version name
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "unyt",
        "pycorr",
    ],
)
