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

"""Simple classification of synthetic data with Fisher-Rao norm regularization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import direct_fisher_rao as dfr
import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(
    "regularization_norm",
    default="fr",
    help=("Use None for no regularization, fr for standard Fisher-Rao norm,"
          " and fre for empirical Fisher-Rao norm."))
tf.flags.DEFINE_float(
    "fisher_rao_lambda",
    default=0,
    help=("Factor with which Fisher-Rao regularization loss term enters the"
          " total loss."))
tf.flags.DEFINE_bool(
    "differentiate_probability",
    default=True,
    help=("If the standard Fisher-Rao norm is chosen, this flag determines"
          " whether the probability term of the categorical expectation should"
          " enter the gradient computation."))
tf.flags.DEFINE_string(
    "optimizer",
    default="sgd",
    help=("Optimizer to be used for training, either sgd or adam."))
tf.flags.DEFINE_float(
    "learning_rate",
    default=1e-3,
    help=("Learning rate to be used for the optimizer."))
tf.flags.DEFINE_integer(
    "train_steps",
    default=100001,
    help=("Number of steps to train for."))
tf.flags.DEFINE_integer(
    "eval_steps",
    default=2000,
    help=("Number of steps between evaluating accuracy and loss on the full"
          " training set."))
tf.flags.DEFINE_integer(
    "batch_size",
    default=64,
    help=("Batch size to be used for training."))
tf.flags.DEFINE_integer(
    "train_size",
    default=5000,
    help=("Size of the randomly generated training set."))
tf.flags.DEFINE_integer(
    "test_size",
    default=1000,
    help=("Size of the randomly generated test set."))
tf.flags.DEFINE_integer(
    "num_hidden_layers",
    default=3,
    help=("Number of hidden layers."))
tf.flags.DEFINE_integer(
    "layer_width",
    default=10,
    help=("Width of hidden layers."))
tf.flags.DEFINE_integer(
    "random_seed",
    default=1,
    help=("Random seed for both generated data and initial weights."))
tf.flags.DEFINE_integer(
    "input_dimension",
    default=10,
    help=("Input dimension for randomly generated data set."))
tf.flags.DEFINE_string(
    "output_dir",
    default="/tmp/",
    help=("File system location for writing output file."))


OUTPUT_FLAGS = ["regularization_norm", "fisher_rao_lambda",
                "differentiate_probability", "optimizer",
                "learning_rate", "train_steps", "eval_steps",
                "batch_size", "train_size", "test_size",
                "num_hidden_layers", "layer_width", "random_seed",
                "input_dimension"]


def make_output_flags_string():
  """Returns a string that encodes the hparam flags that are being used."""
  flags_string = "[" + str(int(time.time()))
  for flag in OUTPUT_FLAGS:
    flags_string += ", \"" + str(getattr(FLAGS, flag)) + "\""

  flags_string += "]"

  return flags_string


def make_output_filename():
  """Returns a filename that encodes the hparam flags that are being used."""
  output_filename = str(int(time.time()))
  for flag in OUTPUT_FLAGS:
    output_filename += "_" + str(getattr(FLAGS, flag))

  return output_filename


def accuracy_reduction(labels, predictions):
  """Computes accuracy metric for class labels given prediction probabilities.

  Args:
    labels: Tensor, contains the one hot encoding of class labels.
    predictions: Tensor, contains the prediction probabilities, should have the
      same shape as labels.

  Returns:
    Scalar, the ratio of correctly predicted labels.
  """
  correct_prediction = tf.equal(tf.round(predictions), tf.round(labels))
  count_correct_predictions = tf.reduce_all(correct_prediction, axis=-1)
  return tf.reduce_mean(tf.cast(count_correct_predictions, tf.float32))


def main(unused_argv):
  del unused_argv

  tf.set_random_seed(FLAGS.random_seed)
  sess = tf.Session()

  # Build datasets
  inputs = tf.random_normal((FLAGS.train_size + FLAGS.test_size,
                             FLAGS.input_dimension))
  weights = tf.random_normal((FLAGS.input_dimension, 2))
  logits = tf.matmul(inputs, weights)
  distribution = tf.distributions.Categorical(logits=sess.run(logits))
  labels = distribution.sample()
  labels = tf.one_hot(labels, 2)

  train_data = tf.data.Dataset.from_tensor_slices((sess.run(inputs),
                                                   sess.run(labels)))
  train_data = train_data.take(FLAGS.train_size)
  train_data = train_data.cache()
  train_data = train_data.repeat()

  train_data_batch = train_data.batch(FLAGS.batch_size)
  train_data_batch_next = train_data_batch.make_one_shot_iterator().get_next()

  train_data = train_data.batch(FLAGS.train_size)
  train_data_next = train_data.make_one_shot_iterator().get_next()

  test_data = tf.data.Dataset.from_tensor_slices((sess.run(inputs),
                                                  sess.run(labels)))
  test_data = test_data.skip(FLAGS.train_size)
  test_data = test_data.take(FLAGS.test_size)
  test_data = test_data.cache()
  test_data = test_data.repeat()

  test_data = test_data.batch(FLAGS.test_size)
  test_data_next = test_data.make_one_shot_iterator().get_next()

  # Build graph
  input_batch = tf.placeholder(tf.float32, shape=(None, FLAGS.input_dimension))
  label_batch = tf.placeholder(tf.float32, shape=(None, 2))

  def make_logits():
    """Builds fully connected ReLU neural network model and returns logits."""
    input_layer = tf.layers.dense(inputs=input_batch,
                                  units=FLAGS.layer_width,
                                  activation=tf.nn.relu)
    previous_layer = input_layer

    for _ in range(FLAGS.num_hidden_layers):
      layer = tf.layers.dense(inputs=previous_layer,
                              units=FLAGS.layer_width,
                              activation=tf.nn.relu)
      previous_layer = layer

    logits = tf.layers.dense(inputs=previous_layer, units=2)

    return logits

  if FLAGS.regularization_norm == "None":
    with tf.variable_scope("regularizer_scope"):
      logits = make_logits()
    regularizer = tf.constant(0.)

  elif FLAGS.regularization_norm == "fre":
    logits, regularizer = dfr.make_empirical_fisher_regularizer(
        make_logits,
        label_batch,
        "regularizer_scope",
        lambda name: True,
        1e-4)

  elif FLAGS.regularization_norm == "fr":
    logits, regularizer = dfr.make_standard_fisher_regularizer(
        make_logits,
        "regularizer_scope",
        lambda name: True,
        1e-4,
        FLAGS.differentiate_probability)

  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_batch,
                                                 logits=logits))
  total_loss = loss + FLAGS.fisher_rao_lambda * regularizer

  if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

  elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  train = optimizer.minimize(total_loss)

  accuracy = accuracy_reduction(labels=label_batch,
                                predictions=tf.nn.softmax(logits))

  train_loss_trajectory = []
  train_accuracy_trajectory = []

  test_loss_trajectory = []
  test_accuracy_trajectory = []

  regularizer_trajectory = []

  sess.run(tf.global_variables_initializer())

  # Optimization loop
  for i in range(FLAGS.train_steps):
    if i % FLAGS.eval_steps == 0:
      tf.logging.info("iter " + str(i) + " / " + str(FLAGS.train_steps))

      # Train loss and accuracy
      train_sample = sess.run(train_data_next)

      train_loss = sess.run(tf.reduce_mean(loss),
                            feed_dict={input_batch: train_sample[0],
                                       label_batch: train_sample[1]})
      train_loss_trajectory.append(train_loss)

      train_accuracy = sess.run(accuracy,
                                feed_dict={input_batch: train_sample[0],
                                           label_batch: train_sample[1]})
      train_accuracy_trajectory.append(train_accuracy)

      tf.logging.info("train loss " + str(train_loss))
      tf.logging.info("train accuracy " + str(train_accuracy))

      # Test loss and accuracy
      test_sample = sess.run(test_data_next)

      test_loss = sess.run(tf.reduce_mean(loss),
                           feed_dict={input_batch: test_sample[0],
                                      label_batch: test_sample[1]})
      test_loss_trajectory.append(test_loss)

      test_accuracy = sess.run(accuracy,
                               feed_dict={input_batch: test_sample[0],
                                          label_batch: test_sample[1]})
      test_accuracy_trajectory.append(test_accuracy)

      tf.logging.info("test loss " + str(test_loss))
      tf.logging.info("test accuracy " + str(test_accuracy))

    batch_sample = sess.run(train_data_batch_next)

    regularizer_loss, _ = sess.run((regularizer, train),
                                   feed_dict={input_batch: batch_sample[0],
                                              label_batch: batch_sample[1]})
    regularizer_trajectory.append(regularizer_loss)

  output_filename = FLAGS.output_dir + make_output_filename()
  with tf.gfile.Open(output_filename, "w") as output_file:
    output_file.write("{\n")
    output_file.write("\"hparams\" : " + make_output_flags_string() + ",\n")
    output_file.write("\"train_loss\" : " + str(train_loss_trajectory) + ",\n")
    output_file.write(("\"train_accuracy\" : " + str(train_accuracy_trajectory)
                       + ",\n"))
    output_file.write("\"test_loss\" : " + str(test_loss_trajectory) + ",\n")
    output_file.write(("\"test_accuracy\" : " + str(test_accuracy_trajectory)
                       + ",\n"))
    output_file.write(("\"regularizer_loss\" : " + str(regularizer_trajectory)
                       + "\n"))
    output_file.write("}\n")

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
