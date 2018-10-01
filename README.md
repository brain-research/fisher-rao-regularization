# Fisher-Rao Regularization

Disclaimer: This is not an official Google product.

## Introduction

The goal of the Fisher-Rao regularization project is to evaluate the
performance of the Fisher-Rao norm proposed in (Liang et al. 2017) as an
explicit regularizer.

This repository contains an implementation of the Fisher-Rao regularizer loss
term based on mini-batch approximations of the standard and empirical Fisher
information matrices.

## Required packages:

- Tensorflow

## Quickstart:

To train the synthetic data model:

```console
python3 synthetic_classification.py
```

By default the model outputs results to the folder `/tmp`.
If you have other files in that folder you can set the `output_dir`
flag to something else.
For help in setting flags and various hyper parameters:

```console
python3 synthetic_classification.py --help
```

To visualize the results:

```console
python3 plot_utils.py
```

The directory to read results from is set in `plot_utils.py` via the line
```console
  data = load_data_for_pattern(pattern, "/tmp/")
```

## Integration:

To use the Fisher-Rao regularizer for your own classifier, all you need is a
function that constructs the models' TensorFlow graph and returns the logits
Tensor.
The regularizer can then capture variables using the following codes.

For regularization with the standard Fisher information matrix:

```python
import direct_fisher_rao as dfr

logits, regularizer = dfr.make_standard_fisher_regularizer(
    make_logits,                          # Function that builds network graph
    "fisher_rao",                         # Name of VariableScope
    lambda name: "embedding" not in name, # Which variables to regularize
    1e-4,                                 # Finite difference perturbation
    True)                                 # Differentiate probability

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                  logits=logits)
total_loss = tf.reduce_mean(loss) + FLAGS.fisher_rao_lambda * regularizer
```

For regularization with the empirical Fisher information matrix:

```python
import direct_fisher_rao as dfr

logits, regularizer = dfr.make_empirical_fisher_regularizer(
    make_logits,                          # Function that builds network graph
    labels,                               # Labels with same dimension as logits
    "fisher_rao",                         # Name of VariableScope
    lambda name: "embedding" not in name, # Which variables to regularize
    1e-4)                                 # Finite difference perturbation

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                  logits=logits)
total_loss = tf.reduce_mean(loss) + FLAGS.fisher_rao_lambda * regularizer
```

