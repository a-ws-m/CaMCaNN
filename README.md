# CaMCaNN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

Source code and trained models for the paper *Analyzing the Accuracy of Critical Micelle Concentration Predictions using Deep Learning*.

## Installation

Clone this repository and then use `conda` to install the required dependencies.  Next, install the appropriate version of `tensorflow-probability` for your version of `tensorflow`, consulting the [GPFlow installation instructions](https://gpflow.github.io/GPflow/develop/installation.html). (This will [depend on which version of CUDA you have installed](https://www.tensorflow.org/install/source#gpu), if you plan to use GPU acceleration.). Use `pip` to install `gpflow`, `spektral` and `keras_tuner`, and then install the source code of the repository.
```bash
export TFP_VERSION=0.18.*
git clone https://github.com/a-ws-m/CaMCaNN.git
cd CaMCaNN
conda env create -n camcann --file camcann.yml
conda activate camcann
pip install spektral gpflow keras_tuner tensorflow-probability==$TFP_VERSION
pip install -e .
```

## Running experiments

The experiments from the paper can be executed by running the `camcann.lab` module with the appropriate arguments:
```
$ python -m camcann.lab -h
usage: lab.py [-h] [-e EPOCHS] [--cluster] [--and-uq] [--just-uq] [--no-gp-scaler] [--lin-mean-fn] [--only-best] [--test-complementary] [--kpca KPCA] [--pairwise] [--splits SPLITS] [--repeats REPEATS] {GNNModel,ECFPLinear} {Nonionics,All} name

positional arguments:
  {GNNModel,ECFPLinear}
                        The type of model to create.
  {Nonionics,All}       The dataset to use.
  name                  The name of the model.

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to train.
  --cluster             Just perform clustering with the ECFPs.
  --and-uq              Re-train the GNN and then the uncertainty quantification model.
  --just-uq             Just train the uncertainty quantifier.
  --no-gp-scaler        Don't use a scaler on the latent points for the Gaussian process.
  --lin-mean-fn         Use a linear function for the mean of the Gaussian process. If not set, will use the trained MLP from the NN as the mean function.
  --only-best           For GNN -- don't perform search, only train the best model.
  --test-complementary  Test saved model on Complementary data.
  --kpca KPCA           Compute N kernel principal components on Qin and Complementary data after training UQ.
  --pairwise            Compute just the learned pariwise kernel on all of the data.

Sensitivity analysis:
  Flags to set for sensitivity analysis. If unspecified, trains a model using the Qin data split. Otherwise, uses repeated stratified k-fold CV to train models.

  --splits SPLITS       The number of splits to use. Defines the train/test ratio.
  --repeats REPEATS     The number of repeats.
```
The model, its checkpoints and its logs will be saved in `models/<name>`.

For example, to train a linear model using ECFP fingerprints on the whole Qin dataset, you can use:
```bash
python -m camcann.lab ECFPLinear All ecfp-test-all
```

The module is designed to determine the best combination of hyperparameters for training the GNNs using Hyperband. Performing this search is the default behaviour when the model is `GNNModel`. Once at least a single trial has been completed, the `--only-best` flag will train a model with the number of epochs specified with `-e`. After this model has been trained it can supply the latent space inputs for the uncertainty quantification model: `--just-uq` will train the Gaussian process.

For performing sensitivity analysis, the `--cluster` flag must first be used in order to split the data into classes for stratified K-fold cross-validation. Thereafter, the `--splits` and `--repeats` options determine the sensitivity analysis behaviour. Each split/repeat will result in a new folder in the `models` directory.

## Research results

All of the models that were trained during the research are available in the [models](models) subdirectory. The [README](models/README.md) provides a description of each model and the metrics that are available.
