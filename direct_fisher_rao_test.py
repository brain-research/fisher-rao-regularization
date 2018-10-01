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

"""Tests for direct_fisher_rao."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import direct_fisher_rao as dfr
import numpy as np
import tensorflow as tf


class DirectFisherRaoTest(tf.test.TestCase):
  perturbation = 1e-4
  rtol = 1e-2

  def test_empirical_fisher_constant_loss(self):
    """Asserts unregularized loss without variables evaluates to constant."""
    labels = tf.constant([[1.0, 0.0]])
    make_logits = lambda: tf.constant([[0.5, 0.5]])
    logits, _ = dfr.make_empirical_fisher_regularizer(make_logits,
                                                      labels,
                                                      "test_scope",
                                                      lambda name: True,
                                                      self.perturbation)

    with self.test_session() as sess:
      self.assertAllEqual(sess.run(logits), [[0.5, 0.5]])

  def test_empirical_fisher_constant_loss_regularizer(self):
    """Asserts regularizer for loss without variables evaluates to zero."""
    labels = tf.constant([[1.0, 0.0]])
    make_logits = lambda: tf.constant([[0.5, 0.5]])
    _, regularizer = dfr.make_empirical_fisher_regularizer(make_logits,
                                                           labels,
                                                           "test_scope",
                                                           lambda name: True,
                                                           self.perturbation)

    with self.test_session() as sess:
      self.assertAllEqual(sess.run(regularizer), 0.)

  def test_empirical_fisher_should_regularize_unchanged_loss(self):
    """Asserts unregularized loss unchanged by `should_regularize` function."""
    labels = tf.constant([[1.0, 0.0]])
    def make_logits():
      l = tf.get_variable("a", initializer=tf.constant(1.))
      return tf.stack([[l, tf.subtract(1., l)]])

    loss_true, _ = dfr.make_empirical_fisher_regularizer(make_logits,
                                                         labels,
                                                         "test_scope",
                                                         lambda name: True,
                                                         self.perturbation)

    loss_false, _ = dfr.make_empirical_fisher_regularizer(make_logits,
                                                          labels,
                                                          "test_scope_2",
                                                          lambda name: False,
                                                          self.perturbation)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(sess.run(loss_true), sess.run(loss_false))

  def test_empirical_fisher_should_regularize_zero_regularizer(self):
    """Asserts regularizer forced unchanged by `should_regularize` function."""
    labels = tf.constant([[1.0, 0.0]])
    def make_logits():
      l = tf.get_variable("a", initializer=tf.constant(1.))
      return tf.stack([[l, tf.subtract(1., l)]])

    _, regularizer = dfr.make_empirical_fisher_regularizer(make_logits,
                                                           labels,
                                                           "test_scope",
                                                           lambda name: False,
                                                           self.perturbation)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(sess.run(regularizer), 0.0)

  def test_empirical_fisher_should_regularize_changed_regularizer(self):
    """Asserts regularizer correctly changed by `should_regularize` function."""
    labels = tf.constant([[1.0, 0.0]])
    def make_logits():
      a = tf.get_variable("a", initializer=tf.constant(1.))
      b = tf.get_variable("b", initializer=tf.constant(1.))
      l = tf.multiply(a, b)
      return tf.stack([[l, tf.subtract(1., l)]])

    _, regularizer_b = dfr.make_empirical_fisher_regularizer(
        make_logits,
        labels,
        "test_scope_should",
        lambda name: "b" in name,
        # Note that for the "b" in name check to work with the intended effect
        # the scope name cannot contain the letter b
        self.perturbation)

    _, regularizer = dfr.make_empirical_fisher_regularizer(
        make_logits,
        labels,
        "test_scope",
        lambda name: True,
        self.perturbation)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertNotEqual(sess.run(regularizer_b), sess.run(regularizer))

  def make_empirical_fisher_sin_logits_and_regularizer(self):
    """Helper that creates Tensors for sin(x) logits and the regularizer."""
    labels = tf.constant([[1.0, 0.0]])
    def make_logits():
      x = tf.get_variable("x", initializer=tf.constant(2.))
      y = tf.get_variable("y", initializer=tf.constant(3.))
      return tf.stack([[tf.sin(x), tf.sin(y)]])

    return dfr.make_empirical_fisher_regularizer(make_logits,
                                                 labels,
                                                 "test_scope",
                                                 lambda name: True,
                                                 self.perturbation)

  def test_empirical_fisher_sin_logits(self):
    """Asserts unregularized loss for sin logits evaluates symbolic solution."""
    symbolic_logits = [[np.sin(2.), np.sin(3.)]]
    logits, _ = self.make_empirical_fisher_sin_logits_and_regularizer()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(logits), symbolic_logits, rtol=self.rtol)

  def symbolic_regularizer_empirical_fisher_sin_logits(self, x):
    expsum = np.exp(np.sin(2.)) + np.exp(np.sin(x))
    dlogp1_1 = -np.exp(np.sin(x)) * np.cos(2.) / expsum
    dlogp1_2 = np.exp(np.sin(x)) * np.cos(x) / expsum

    symbolic_regularizer = np.square(dlogp1_1 * 2. + dlogp1_2 * x)

    return symbolic_regularizer

  def test_empirical_fisher_sin_regularizer(self):
    """Asserts regularizer for sin logits evaluates symbolic solution."""
    symbolic_regularizer = \
        self.symbolic_regularizer_empirical_fisher_sin_logits(3.)
    _, regularizer = self.make_empirical_fisher_sin_logits_and_regularizer()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(regularizer), symbolic_regularizer,
                          rtol=self.rtol)

  def test_empirical_fisher_sin_gradient(self):
    """Validates gradient of regularizer for sin logits by finite difference."""
    h = 1e-4
    symbolic_regularizer_plus_pert = \
        self.symbolic_regularizer_empirical_fisher_sin_logits(3. + h)
    symbolic_regularizer_minus_pert = \
        self.symbolic_regularizer_empirical_fisher_sin_logits(3. - h)
    fd_gradient = (symbolic_regularizer_plus_pert -
                   symbolic_regularizer_minus_pert) / (2. * h)

    _, regularizer = self.make_empirical_fisher_sin_logits_and_regularizer()

    with tf.variable_scope("test_scope", reuse=True):
      gradient = tf.gradients(regularizer, tf.get_variable("y"))[0]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(gradient),
                          fd_gradient,
                          rtol=np.sqrt(self.rtol))
      # Using sqrt of usual tolerance because two finite difference
      # approximations are being compared

  def make_two_vars_product_loss_and_regularizer(self):
    """Helper that creates Tensors for a * b loss and its regularizer."""
    def make_loss():
      a = tf.get_variable("a", initializer=tf.constant(2.))
      b = tf.get_variable("b", initializer=tf.constant(3.))
      return tf.multiply(a, b)

    return dfr.make_empirical_fisher_regularizer(make_loss,
                                                 "test_scope",
                                                 lambda name: True,
                                                 self.perturbation)

  def test_empirical_fisher_batch(self):
    """Asserts sum property of regularizer gradient for sum reduction loss."""
    labels_full_batch = \
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    labels_one_batch = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    labels_two_batch = [[0.5, 0.5], [0.5, 0.5]]

    x_full_batch = np.array([1., 2., 3., 4., 5.])
    x_part_one_batch = np.array([1., 2., 3.])
    x_part_two_batch = np.array([4., 5.])

    def make_make_logits_part(x_batch):
      def make_logits():
        v = tf.get_variable("v", initializer=tf.constant(2.5))
        y = tf.multiply(v, x_batch)
        return tf.transpose(tf.stack([tf.sin(y), tf.cos(y)]))

      return make_logits

    _, regularizer_full = dfr.make_empirical_fisher_regularizer(
        make_make_logits_part(x_full_batch),
        labels_full_batch,
        "test_scope",
        lambda name: True,
        self.perturbation)

    _, regularizer_part_one = dfr.make_empirical_fisher_regularizer(
        make_make_logits_part(x_part_one_batch),
        labels_one_batch,
        "test_scope_part_one",
        lambda name: True,
        self.perturbation)

    _, regularizer_part_two = dfr.make_empirical_fisher_regularizer(
        make_make_logits_part(x_part_two_batch),
        labels_two_batch,
        "test_scope_part_two",
        lambda name: True,
        self.perturbation)

    with tf.variable_scope("test_scope", reuse=True):
      gradient_full = tf.gradients(regularizer_full, tf.get_variable("v"))[0]

    with tf.variable_scope("test_scope_part_one", reuse=True):
      gradient_part_one = tf.gradients(regularizer_part_one,
                                       tf.get_variable("v"))[0]

    with tf.variable_scope("test_scope_part_two", reuse=True):
      gradient_part_two = tf.gradients(regularizer_part_two,
                                       tf.get_variable("v"))[0]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(
          5 * sess.run(gradient_full),
          3 * sess.run(gradient_part_one) + 2 * sess.run(gradient_part_two),
          rtol=self.rtol)

  def helper_standard_fisher_constant_loss(self, differentiate_probability):
    """Asserts logits without variables evaluates to constant."""
    make_logits = lambda: tf.constant([[0.5, 0.5]])

    logits, _ = dfr.make_standard_fisher_regularizer(make_logits,
                                                     "test_scope",
                                                     lambda name: True,
                                                     self.perturbation,
                                                     differentiate_probability)

    with self.test_session() as sess:
      self.assertAllEqual(sess.run(logits), [[0.5, 0.5]])

  def test_standard_fisher_constant_loss(self):
    self.helper_standard_fisher_constant_loss(True)

  def test_standard_fisher_constant_loss_stop(self):
    self.helper_standard_fisher_constant_loss(False)

  def helper_standard_fisher_constant_regularizer(self,
                                                  differentiate_probability):
    """Asserts regularizer for logits without variables evaluates to zero."""
    make_logits = lambda: tf.constant([[0.5, 0.5]])

    _, regularizer = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope",
        lambda name: True,
        self.perturbation,
        differentiate_probability)

    with self.test_session() as sess:
      self.assertAllEqual(sess.run(regularizer), 0.)

  def test_standard_fisher_constant_regularizer(self):
    self.helper_standard_fisher_constant_regularizer(True)

  def test_standard_fisher_constant_regularizer_stop(self):
    self.helper_standard_fisher_constant_regularizer(False)

  def helper_standard_fisher_should_regularize_unchanged_loss(
      self,
      differentiate_probability):
    """Asserts unregularized loss unchanged by `should_regularize` function."""
    def make_logits():
      l = tf.get_variable("a", initializer=tf.constant(1.))
      return tf.stack([[l, tf.subtract(1., l)]])

    loss_true, _ = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope_true",
        lambda name: True,
        self.perturbation,
        differentiate_probability)

    loss_false, _ = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope_false",
        lambda name: False,
        self.perturbation,
        differentiate_probability)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(sess.run(loss_true), sess.run(loss_false))

  def test_standard_fisher_should_regularize_unchanged_loss(self):
    self.helper_standard_fisher_should_regularize_unchanged_loss(True)

  def test_standard_fisher_should_regularize_unchanged_loss_stop(self):
    self.helper_standard_fisher_should_regularize_unchanged_loss(False)

  def helper_standard_fisher_should_regularize_zero_regularizer(
      self,
      differentiate_probability):
    """Asserts regularizer forced unchanged by `should_regularize` function."""
    def make_logits():
      l = tf.get_variable("a", initializer=tf.constant(1.))
      return tf.stack([[l, tf.subtract(1., l)]])

    _, regularizer = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope",
        lambda name: False,
        self.perturbation,
        differentiate_probability)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(sess.run(regularizer), 0.0)

  def test_standard_fisher_should_regularize_zero_regularizer(self):
    self.helper_standard_fisher_should_regularize_zero_regularizer(True)

  def test_standard_fisher_should_regularize_zero_regularizer_stop(self):
    self.helper_standard_fisher_should_regularize_zero_regularizer(False)

  def helper_standard_fisher_should_regularize_different_regularizer(
      self,
      differentiate_probability):
    """Asserts regularizer changed by `should_regularize` function."""
    def make_logits():
      a = tf.get_variable("a", initializer=tf.constant(1.))
      b = tf.get_variable("b", initializer=tf.constant(1.))
      l = tf.multiply(a, b)
      return tf.stack([[l, tf.subtract(1., l)]])

    _, regularizer_b = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope_should",
        lambda name: "b" in name,
        self.perturbation,
        differentiate_probability)

    _, regularizer = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope",
        lambda name: True,
        self.perturbation,
        differentiate_probability)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertNotEqual(sess.run(regularizer_b), sess.run(regularizer))

  def test_standard_fisher_should_regularize_different_regularizer(self):
    self.helper_standard_fisher_should_regularize_different_regularizer(True)

  def test_standard_fisher_should_regularize_different_regularizer_stop(self):
    self.helper_standard_fisher_should_regularize_different_regularizer(False)

  def helper_standard_fisher_should_regularize_symmetric_regularizer(
      self,
      differentiate_probability):
    """Asserts regularizer changed by `should_regularize` function."""
    def make_logits():
      a = tf.get_variable("a", initializer=tf.constant(1.))
      b = tf.get_variable("b", initializer=tf.constant(1.))
      l = tf.multiply(a, b)
      return tf.stack([[l, tf.subtract(1., l)]])

    _, regularizer_b = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope_should",
        lambda name: "b" in name,
        self.perturbation,
        differentiate_probability)

    _, regularizer_a = dfr.make_standard_fisher_regularizer(
        make_logits,
        "test_scope_should_symmetric",
        lambda name: "a" in name,
        self.perturbation,
        differentiate_probability)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(sess.run(regularizer_b), sess.run(regularizer_a))

  def test_standard_fisher_should_regularize_symmetric_regularizer(self):
    self.helper_standard_fisher_should_regularize_symmetric_regularizer(True)

  def test_standard_fisher_should_regularize_symmetric_regularizer_stop(self):
    self.helper_standard_fisher_should_regularize_symmetric_regularizer(False)

  def make_standard_fisher_sin_logits_and_regularizer(
      self,
      differentiate_probability):
    """Helper that creates Tensors for sin(x) logits and the regularizer."""
    def make_logits():
      x = tf.get_variable("x", initializer=tf.constant(2.))
      y = tf.get_variable("y", initializer=tf.constant(3.))
      return tf.stack([[tf.sin(x), tf.sin(y)]])

    return dfr.make_standard_fisher_regularizer(make_logits,
                                                "test_scope",
                                                lambda name: True,
                                                self.perturbation,
                                                differentiate_probability)

  def helper_standard_fisher_sin_logits(self, differentiate_probability):
    """Asserts unregularized loss for sin logits evaluates symbolic solution."""
    symbolic_logits = [[np.sin(2.), np.sin(3.)]]
    logits, _ = self.make_standard_fisher_sin_logits_and_regularizer(
        differentiate_probability)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(logits), symbolic_logits, rtol=self.rtol)

  def test_standard_fisher_sin_logits(self):
    self.helper_standard_fisher_sin_logits(True)

  def test_standard_fisher_sin_logits_stop(self):
    self.helper_standard_fisher_sin_logits(False)

  def symbolic_regularizer_standard_fisher_sin_logits_no_probs(self, x, p1, p2):
    expsum = np.exp(np.sin(2.)) + np.exp(np.sin(x))
    dlogp1_1 = np.cos(2.) - 1. / expsum * np.exp(np.sin(2.)) * np.cos(2.)
    dlogp1_2 = - 1. / expsum * np.exp(np.sin(x)) * np.cos(x)
    dlogp2_1 = - 1. / expsum * np.exp(np.sin(2.)) * np.cos(2.)
    dlogp2_2 = np.cos(x) - 1. / expsum * np.exp(np.sin(x)) * np.cos(x)

    symbolic_regularizer = (np.square(dlogp1_1 * 2. + dlogp1_2 * x) * p1 +
                            np.square(dlogp2_1 * 2. + dlogp2_2 * x) * p2)

    return symbolic_regularizer

  def symbolic_regularizer_standard_fisher_sin_logits(self, x):
    expsum = np.exp(np.sin(2.)) + np.exp(np.sin(x))
    p1 = np.exp(np.sin(2.)) / expsum
    p2 = np.exp(np.sin(x)) / expsum

    return self.symbolic_regularizer_standard_fisher_sin_logits_no_probs(x,
                                                                         p1,
                                                                         p2)

  def helper_standard_fisher_sin_regularizer(self, differentiate_probability):
    """Asserts regularizer for sin logits evaluates symbolic solution."""
    symbolic_regularizer = \
        self.symbolic_regularizer_standard_fisher_sin_logits(3.)
    _, regularizer = self.make_standard_fisher_sin_logits_and_regularizer(
        differentiate_probability)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(regularizer), symbolic_regularizer,
                          rtol=self.rtol)

  def test_standard_fisher_sin_regularizer(self):
    self.helper_standard_fisher_sin_regularizer(True)

  def test_standard_fisher_sin_regularizer_stop(self):
    self.helper_standard_fisher_sin_regularizer(False)

  def test_standard_fisher_sin_regularizer_gradient(self):
    """Validates gradient of regularizer for sin logits by finite difference."""
    h = 1e-4
    symbolic_regularizer_pert = \
        self.symbolic_regularizer_standard_fisher_sin_logits(3. + h)
    symbolic_regularizer = \
        self.symbolic_regularizer_standard_fisher_sin_logits(3.)
    fd_gradient = (symbolic_regularizer_pert - symbolic_regularizer) / h

    _, regularizer = self.make_standard_fisher_sin_logits_and_regularizer(True)

    with tf.variable_scope("test_scope", reuse=True):
      gradient = tf.gradients(regularizer, tf.get_variable("y"))[0]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(gradient), fd_gradient, rtol=self.rtol)

  def test_standard_fisher_sin_regularizer_gradient_stop(self):
    """Validates gradient of regularizer for sin logits by finite difference."""
    expsum = np.exp(np.sin(2.)) + np.exp(np.sin(3.))
    p1 = np.exp(np.sin(2.)) / expsum
    p2 = np.exp(np.sin(3.)) / expsum

    h = 1e-4
    symbolic_regularizer_pert = \
        self.symbolic_regularizer_standard_fisher_sin_logits_no_probs(3. + h,
                                                                      p1,
                                                                      p2)
    symbolic_regularizer = \
        self.symbolic_regularizer_standard_fisher_sin_logits_no_probs(3.,
                                                                      p1,
                                                                      p2)
    fd_gradient = (symbolic_regularizer_pert - symbolic_regularizer) / h

    _, regularizer = self.make_standard_fisher_sin_logits_and_regularizer(False)

    with tf.variable_scope("test_scope", reuse=True):
      gradient = tf.gradients(regularizer, tf.get_variable("y"))[0]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(gradient), fd_gradient, rtol=self.rtol)

  def helper_standard_fisher_categorical_batch(self, differentiate_probability):
    """Asserts sum property of regularizer gradient for sum reduction loss."""
    x_full_batch = np.array([1., 2., 3., 4., 5.])
    x_part_one_batch = np.array([1., 2., 3.])
    x_part_two_batch = np.array([4., 5.])

    def make_make_logits_part(x_batch):
      def make_logits():
        v = tf.get_variable("v", initializer=tf.constant(2.5))
        y = tf.multiply(v, x_batch)
        return tf.transpose(tf.stack([tf.sin(y), tf.cos(y)]))

      return make_logits

    _, regularizer_full = dfr.make_standard_fisher_regularizer(
        make_make_logits_part(x_full_batch),
        "test_scope",
        lambda name: True,
        self.perturbation,
        differentiate_probability)

    _, regularizer_part_one = dfr.make_standard_fisher_regularizer(
        make_make_logits_part(x_part_one_batch),
        "test_scope_part_one",
        lambda name: True,
        self.perturbation,
        differentiate_probability)

    _, regularizer_part_two = dfr.make_standard_fisher_regularizer(
        make_make_logits_part(x_part_two_batch),
        "test_scope_part_two",
        lambda name: True,
        self.perturbation,
        differentiate_probability)

    with tf.variable_scope("test_scope", reuse=True):
      gradient_full = tf.gradients(regularizer_full, tf.get_variable("v"))[0]

    with tf.variable_scope("test_scope_part_one", reuse=True):
      gradient_part_one = tf.gradients(regularizer_part_one,
                                       tf.get_variable("v"))[0]

    with tf.variable_scope("test_scope_part_two", reuse=True):
      gradient_part_two = tf.gradients(regularizer_part_two,
                                       tf.get_variable("v"))[0]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(
          5 * sess.run(gradient_full),
          3 * sess.run(gradient_part_one) + 2 * sess.run(gradient_part_two),
          rtol=self.rtol)

  def test_standard_fisher_categorical_batch(self):
    self.helper_standard_fisher_categorical_batch(True)

  def test_standard_fisher_categorical_batch_stop(self):
    self.helper_standard_fisher_categorical_batch(False)

if __name__ == "__main__":
  tf.test.main()
