"""Setup script for package."""
import pathlib
from setuptools import find_namespace_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="CaMCaNN",
    version="0.0.1",
    description="Group contribution methods for CMC prediction.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/a-ws-m/camcann",
    author="Alexander Moriarty",
    author_email="amoriarty14@gmail.com",
    license="MIT",
    keywords=[
        "keras",
        "tensorflow",
        "critical micelle concentration",
        "surfactants",
        "machine learning",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    packages=find_namespace_packages(include=["camcann*"]),
    include_package_data=True,
    install_requires=[
        "rdkit",
        "tensorflow",
        "tensorflow-probability",
        "scikit-learn",
        "pandas",
        "scipy",
        "seaborn",
        "spektral",
        "keras_tuner",
        "gpflow",
    ],
    python_requires=">=3.6",
)
