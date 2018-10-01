# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for plotting results."""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


ALPHA1 = 0.4
ALPHA2 = 0.2
COLOR1 = "red"
COLOR2 = "blue"
COLOR3 = "green"
COLOR4 = "purple"
COLOR5 = "orange"
COLOR6 = "black"


def load_data_for_pattern(pattern, data_dir):
  """Loads data from all files in data_dir that match a regular expression."""

  file_list = os.listdir(data_dir)
  data = []

  for filename in file_list:
    if pattern.match(filename):
      with open(data_dir + filename, "r") as input_file:
        data.append(json.load(input_file))

  return data


def compute_statistics(data, variable):
  """Computes mean and standard deviations for given variable in loaded data."""

  data_mean = data[0][variable]
  data_std = np.zeros(len(data_mean))

  if len(data) > 1:
    for i in range(1, len(data)):
      data_mean = np.add(data_mean, data[i][variable])
    data_mean = np.divide(data_mean, len(data))

    for i in range(len(data)):
      data_std = np.add(data_std,
                        np.square(np.subtract(data[i][variable], data_mean)))
    data_std = np.sqrt(np.divide(data_std, len(data) - 1))

  return (data_mean, data_std)


def draw_plot(data_mean, data_std, label, title, color, factor):
  """Draws plot with two standard deviation error bars."""
  num_iter = np.multiply(factor, range(0, len(data_mean)))

  low_fill = np.subtract(data_mean, np.multiply(2., data_std))
  high_fill = np.add(data_mean, np.multiply(2., data_std))
  plt.plot(num_iter, data_mean, color=color, alpha=ALPHA1, label=label)
  plt.fill_between(num_iter, low_fill, high_fill, color=color, lw=0,
                   alpha=ALPHA2)

  plt.xlabel("# iter")
  plt.ylabel(title)
  plt.legend(loc="best")


def example():
  """Demonstrates how to plot results from two experiments to compare results.
  """

  plt.figure()

  label = ".*"
  pattern = re.compile(label)
  data = load_data_for_pattern(pattern, "/tmp/data/")

  train_loss_mean, train_loss_std = compute_statistics(data, "train_loss")
  train_acc_mean, train_acc_std = compute_statistics(data, "train_accuracy")
  test_loss_mean, test_loss_std = compute_statistics(data, "test_loss")
  test_acc_mean, test_acc_std = compute_statistics(data, "test_accuracy")
  regularizer_mean, regularizer_std = compute_statistics(data,
                                                         "regularizer_loss")
  plt.subplot(1, 3, 1)
  draw_plot(train_loss_mean, train_loss_std, "train", "loss", COLOR1, 100)
  draw_plot(test_loss_mean, test_loss_std, "test", "loss", COLOR2, 100)

  plt.subplot(1, 3, 2)
  draw_plot(train_acc_mean, train_acc_std, "train", "accuracy", COLOR1, 100)
  draw_plot(test_acc_mean, test_acc_std, "test", "accuracy", COLOR2, 100)

  plt.subplot(1, 3, 3)
  draw_plot(regularizer_mean, regularizer_std, "regularizer", "loss", COLOR1, 1)

  plt.show()

if __name__ == '__main__':
  example()
