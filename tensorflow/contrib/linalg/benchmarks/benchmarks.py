import os
import time

import numpy as np
import pandas as pd

from tensorflow.python import convert_to_tensor
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.variables import Variable, global_variables_initializer
from tensorflow.python.ops.math_ops import reduce_sum, square
from tensorflow.python.ops.linalg.linalg import einsum
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.client.session import Session
# from tensorflow.contrib.linalg.bvls import tf_bvls


dir_path = os.path.dirname(os.path.realpath(__file__))
cs_path = os.path.join(dir_path, "benchmarks.csv")


class TestCase(object):
    def __init__(self, n_data, n_vs, noise=0.0):
        self.n_data = n_data
        self.n_vs = n_vs

        self.m = np.random.normal(0.0, 1.0, (n_data, n_vs))
        self.m[:, 1] = self.m[:, 0] * 1E-3 + self.m[:, 1] * 1E-6

        self.lower_bounds = np.random.uniform(-1.0, 0.0, (n_vs,))
        self.upper_bounds = np.random.uniform(0.0, 1.0, (n_vs,))
        self.x = np.random.uniform(self.lower_bounds, self.upper_bounds, (n_vs,))
        self.rhs = np.dot(self.m, self.x) + np.random.normal(0.0, noise, (n_data,))

        self.tf_m = convert_to_tensor(self.m)
        self.tf_rhs = convert_to_tensor(self.rhs)
        self.tf_lower_bounds = convert_to_tensor(self.lower_bounds)
        self.tf_upper_bounds = convert_to_tensor(self.upper_bounds)

        self.algorithm = ""
        self.runtime = 0
        self.solution = np.zeros_like(self.x)
        self.iterations = 0
        self.loss = 0

    @property
    def bias(self):
        residuals = self.x - self.solution
        return np.linalg.norm(residuals) / self.n_vs

    @property
    def n_bounded(self):
        lower = self.solution <= self.lower_bounds
        upper = self.solution >= self.upper_bounds
        return np.sum(lower) + np.sum(upper)

    def save(self, algorithm):
        print("-" * 100)
        print("True x: ", self.x)
        print("Estimated x: ",  self.solution)

        if os.path.exists(cs_path):
            df = pd.read_csv(cs_path)
        else:
            df = pd.DataFrame(
                columns=["Algorithm", "N Data", "N Variables", "Runtime (ms)", "N Bounded", "Loss", "Bias", "Iterations"]
            )

        df = df.append(pd.DataFrame({
            "Algorithm": [algorithm],
            "N Data": [self.n_data],
            "N Variables": [self.n_vs],
            "Runtime (ms)": [self.runtime],
            "N Bounded": [self.n_bounded],
            "Loss": [self.loss],
            "Bias": [self.bias],
            "Iterations": [self.iterations],
        }), ignore_index=True)

        df.to_csv(cs_path, index=False)


def solve_gradient(m, rhs, lower_bounds, upper_bounds):
    nw = m.shape[1]
    w = Variable(
        zeros((nw,), dtype=m.dtype),
        constraint=lambda x: clip_by_value(x, lower_bounds, upper_bounds),
        name="w",
    )

    # Least square residuals
    lstsq_residuals = einsum("ij,j->i", m, w) - rhs

    # Least square loss
    tf_loss = reduce_sum(square(lstsq_residuals))

    # Prior loss
    # tf_loss += reduce_sum(w[0] * w[0])

    train = AdamOptimizer(0.05).minimize(
        tf_loss,
        var_list=[w],
    )
    init = global_variables_initializer()

    with Session() as sess:
        sess.run(init)

        loss_prev = 1E30
        loss = 1E29
        i = 0
        start = time.time()
        while abs(loss - loss_prev) > 1E-16 and i < 5E4:
            i += 1
            loss_prev = loss
            loss, _ = sess.run((tf_loss, train))

        end = time.time()
        runtime = round(1000 * (end - start))
        w_result = sess.run(w)
        loss = sess.run(tf_loss)
        print("Loss ", i, ":", loss, "Time (ms): ", runtime)

        return w_result, loss, runtime, i


def solve_gradient_and_save(test_case):
    w_result, loss, runtime, iterations = solve_gradient(
        test_case.tf_m,
        test_case.tf_rhs,
        test_case.tf_lower_bounds,
        test_case.tf_upper_bounds,
    )

    test_case.solution = w_result
    test_case.runtime = runtime
    test_case.iterations = iterations
    test_case.loss = loss
    test_case.save("AdamOptimizer")


for _ in range(10):
    test_case = TestCase(n_data=5, n_vs=5, noise=1E-2)
    solve_gradient_and_save(test_case)


with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
    print(pd.read_csv(cs_path))
