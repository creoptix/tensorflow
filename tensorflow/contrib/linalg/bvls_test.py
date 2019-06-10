# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for tf.contrib.linalg.bvls."""

import time
import unittest

import numpy as np

from tensorflow.python import convert_to_tensor

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.training import adam

from tensorflow.contrib.linalg.bvls import (
    lstsq_negative_gradient,
    tf_bvls,
    free_variable_with_largest_gradient,
    free_lstsq,
    free_bounded_step,
    tf_bvls_batch,
    lstsq_squared_residuals_sum,
)


class TestTfBvls(unittest.TestCase):
    """
    Test cases for the bounded variable least square solver
    """

    @staticmethod
    def getTestCase(nd=5, nw=4):
        m = np.random.normal(0.0, 1.0, (nd, nw))
        rhs = np.random.normal(0.0, 1.0, (nd,))
        lower_bounds = np.random.uniform(-1.0, 0.0, (nw,))
        upper_bounds = np.random.uniform(0.0, 1.0, (nw,))
        noise_precision = np.random.uniform(1E-2, 1E-1, ())
        prior_precision = np.random.uniform(0., 1E-2, (nw,))
        target_weights = np.random.uniform(0., 1., (nd,))

        m = convert_to_tensor(m)
        rhs = convert_to_tensor(rhs)
        lower_bounds = convert_to_tensor(lower_bounds)
        upper_bounds = convert_to_tensor(upper_bounds)

        return m, rhs, lower_bounds, upper_bounds, noise_precision, prior_precision, target_weights

    @staticmethod
    def solve(
            m,
            rhs,
            lower_bounds,
            upper_bounds,
            noise_precision=None,
            prior_precision=None,
            target_weights=None):

        nw = m.shape[1]

        if noise_precision is None:
            noise_precision = constant_op.constant(1., dtype=m.dtype)

        if prior_precision is None:
            prior_precision = array_ops.zeros(nw, dtype=rhs.dtype)

        if target_weights is None:
            target_weights = array_ops.ones_like(rhs, dtype=rhs.dtype)

        w = variables.Variable(
            array_ops.zeros((nw,), dtype=m.dtype),
            constraint=lambda x: clip_ops.clip_by_value(x, lower_bounds, upper_bounds),
            name="w",
        )

        # Least square residuals
        lstsq_residuals = target_weights * (special_math_ops.einsum("ij,j->i", m, w) - rhs)

        # Least square loss
        tf_loss = noise_precision * math_ops.reduce_sum(math_ops.square(lstsq_residuals))

        # Prior loss
        tf_loss += math_ops.reduce_sum(prior_precision * w * w)

        train = adam.AdamOptimizer(0.01).minimize(
            tf_loss,
            var_list=[w],
        )
        init = variables.global_variables_initializer()

        with session.Session() as sess:
            sess.run(init)

            loss_prev = 1E30
            loss = 1E29
            i = 0
            start = time.time()
            while abs(loss - loss_prev) > 1E-16 and i < 5000:
                i += 1
                loss_prev = loss
                loss, _ = sess.run((tf_loss, train))

            end = time.time()
            w_result = sess.run(w)
            loss = sess.run(tf_loss)
            print("Loss ", i, ":", loss, "Time (ms): ", round(1000 * (end - start)))

            return w_result, loss

    @staticmethod
    def timed_execution(func):
        result = None
        start = time.time()

        for _ in range(100):
            result = func()

        end = time.time()
        execution_time = round(1000 * (end - start) / 100, 3)

        return result, execution_time

    def setUp(self):
        """
        Initialize a random bounded least square regression problem.
        """

        # Bounded regression example
        self.m1 = np.array([
            [0.890197, 0.98748, 0.597844],
            [0.686742, 0.0558757, 0.201711],
            [0.383872, 0.96083, 0.319599],
        ])
        self.rhs1 = np.array([0.360696, 0.945096, 0.106577])

        # Lower bounds
        self.l1 = np.array([-1, -0.5, -1])

        # Upper bounds
        self.u1 = np.array([1, 0.5, 1])

        # Bounded solution
        self.v1 = np.array([1, -0.5, 0.202432])

    def test_bvls_free_variable_with_largest_gradient(self):
        """
        Check that the variable at the boundary with the largest gradient is being freed.
        """

        m1 = ops.convert_to_tensor(self.m1)
        rhs1 = ops.convert_to_tensor(self.rhs1)
        l1 = ops.convert_to_tensor(self.l1)

        # Negative gradient
        n_grad = lstsq_negative_gradient(m1, rhs1, l1)

        lower_mask = [False, True, False]
        upper_mask = [True, False, False]
        tf_result = free_variable_with_largest_gradient(n_grad, lower_mask, upper_mask)

        with session.Session() as sess:
            result = sess.run(tf_result)
            np.testing.assert_equal(result[0], [False, False, False])
            np.testing.assert_equal(result[1], [True, False, False])

    def test_bvls_free_variable_with_largest_gradient_batch(self):
        """
        Check that the variable at the boundary with the largest gradient is being freed.
        """

        m1 = ops.convert_to_tensor(self.m1.reshape((1, 3, 3)))
        rhs1 = ops.convert_to_tensor(self.rhs1.reshape((1, 3)))
        l1 = ops.convert_to_tensor(self.l1.reshape((1, 3)))

        # Negative gradient
        n_grad = lstsq_negative_gradient(m1, rhs1, l1, axis=2)

        lower_mask = [[False, True, False]]
        upper_mask = [[True, False, False]]
        tf_result = free_variable_with_largest_gradient(n_grad, lower_mask, upper_mask)

        with session.Session() as sess:
            result = sess.run(tf_result)
            np.testing.assert_equal(result[0], [[False, False, False]])
            np.testing.assert_equal(result[1], [[True, False, False]])

    def test_bvls_free_lstsq(self):
        """
        Check that least square regression over the free variables works.
        """

        lower_mask = [False, True, False]
        upper_mask = [True, False, False]
        center_mask = [False, False, True]

        m = ops.convert_to_tensor(self.m1)
        rhs = ops.convert_to_tensor(self.rhs1)

        tf_result = free_lstsq(
            m, rhs, center_mask, lower_mask, self.l1, upper_mask, self.u1, fast=False)

        with session.Session() as sess:
            result = sess.run(tf_result)
            np.testing.assert_almost_equal(result, [0, 0, self.v1[-1]], decimal=4)

    def test_bvls_free_bounded_step_passing(self):
        """
        Check that least square regression step respects the bounds.
        """

        lower_mask = [False, True, False]
        upper_mask = [True, False, False]
        center_mask = [False, False, True]

        cvs = np.array([0.0, 0.0, self.v1[-1]], dtype=np.float64)

        tf_result = free_bounded_step(
            center_mask, cvs, lower_mask, self.l1, upper_mask, self.u1)

        with session.Session() as sess:
            result, _ = sess.run(tf_result)
            np.testing.assert_almost_equal(result, [1, -0.5, self.v1[-1]], decimal=4)

    def test_bvls_free_bounded_step_clipped(self):
        """
        Check that least square regression step respects the bounds.
        """

        lower_mask = [False, True, False]
        upper_mask = [True, False, False]
        center_mask = [False, False, True]

        cvs = np.array([0.0, 0.0, 2.0], dtype=np.float64)

        tf_result = free_bounded_step(
            center_mask, cvs, lower_mask, self.l1, upper_mask, self.u1)

        with session.Session() as sess:
            result, _ = sess.run(tf_result)
            np.testing.assert_almost_equal(result, [1, -0.5, 1.0], decimal=4)

    # TODO: write test cases for all boundary scenarios

    def test_bvls(self):
        """
        Check that the bounded least square regression works.
        """

        tf_result = tf_bvls(self.m1, self.rhs1, self.l1, self.u1, fast=False)

        with session.Session() as sess:
            result = sess.run(tf_result)
            print(result)
            np.testing.assert_almost_equal(result, [1, -0.5, self.v1[-1]], decimal=4)

    def test_bvls_batch(self):
        """
        Check that the bounded least square regression works for a batch input.
        """

        m_batch = ops.convert_to_tensor(np.tile(self.m1, (2, 1, 1)))
        rhs_batch = ops.convert_to_tensor(np.tile(self.rhs1, (2, 1)))
        lb_batch = ops.convert_to_tensor(np.tile(self.l1, (2, 1)))
        ub_batch = ops.convert_to_tensor(np.tile(self.u1, (2, 1)))

        tf_result = tf_bvls_batch(m_batch, rhs_batch, lb_batch, ub_batch, fast=False)

        with session.Session() as sess:
            result = sess.run(tf_result)
            print(result)
            # np.testing.assert_almost_equal(result, [1, -0.5, self.v1[-1]], decimal=4)

    # @unittest.skip
    def test_bvls_random_test_cases(self):
        """
        Check that the bounded least square regression works for random test cases.
        """

        for _ in range(10):
            print("-" * 100)

            m, rhs, lb, ub, noise_precision, prior_precision, tws = self.getTestCase(nd=5, nw=3)
            expected_w0, gloss0 = self.solve(m, rhs, lb, ub)
            expected_w, gloss = self.solve(m, rhs, lb, ub, noise_precision, prior_precision, tws)
            tf_bvls_result = tf_bvls(
                m, rhs, lb, ub,
                noise_precision=noise_precision,
                prior_precision=prior_precision,
                target_weights=tws,
                fast=False,
                return_iterations=True,
            )
            tf_lstsq_result = linalg_impl.lstsq(m, array_ops.expand_dims(rhs, -1))
            tf_loss = noise_precision * lstsq_squared_residuals_sum(m, tf_bvls_result[0], rhs, tws)

            with session.Session() as sess:
                (w_result, i), bvls_time = self.timed_execution(lambda: sess.run(tf_bvls_result))
                _, lstsq_time = self.timed_execution(lambda: sess.run(tf_lstsq_result))
                loss = sess.run(tf_loss)

                print("W sample")
                print("Lower bound: ", w_result <= sess.run(lb))
                print("Upper bound: ", w_result >= sess.run(ub))
                print("Result: ", w_result)
                print("Expected0: ", expected_w0)
                print("Expected: ", expected_w)
                print("Assert: ", abs(w_result - expected_w) < 1E-2)
                print("Time (ms): ", round(bvls_time / i, 3), lstsq_time, i)
                print("Loss: ", loss, 100 * (gloss - loss) / loss, "%")

                np.testing.assert_almost_equal(w_result, expected_w, decimal=3)
                self.assertGreaterEqual(gloss, loss)


if __name__ == '__main__':
    unittest.main()
