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
"""Bounded-Variables Least-Squares operations."""

import tensorflow as tf


def tf_bvls_input_validation(matrix, rhs, lower_bounds, upper_bounds):
    """
    Perform basic sanity check on the inputs to the BVLS algorithm.
    :param matrix: Matrix of the least square regression.
    :param rhs: Right hand side of the least square regression.
    :param lower_bounds: Lower bounds of the regression variables.
    :param upper_bounds: Upper bounds on the regression variables.
    """

    # TODO later with tf.assert_shapes
    # assert lower_bounds.shape == upper_bounds.shape
    # assert matrix.shape[-2] == rhs.shape[-1]
    # assert matrix.shape[-1] == lower_bounds.shape[-1]

    # Check that the lower bounds are smaller than the upper bounds
    bounds_check = tf.debugging.assert_less_equal(
        lower_bounds,
        upper_bounds,
        message="Bvls input lower bounds must best less or equal to upper bounds.",
    )

    # Check shapes
    checks = [bounds_check]

    # Add the check to the graph as a control dependency
    with tf.control_dependencies(checks):
        # Check that all values are finite
        matrix = tf.debugging.assert_all_finite(matrix, "Bvls input matrix.")
        rhs = tf.debugging.assert_all_finite(rhs, "Bvls input rhs.")
        lower_bounds = tf.debugging.assert_all_finite(lower_bounds, "Bvls input lower bounds.")
        upper_bounds = tf.debugging.assert_all_finite(upper_bounds, "Bvls input upper bounds.")

        return matrix, rhs, lower_bounds, upper_bounds


def kuhn_tucker_convergence_test_lower(n_grad, variables, lower_bounds):
    """
    Convergence test for the variables at the lower bound.
    The gradient for the variables at the lower bound must be negative.

    :param n_grad: float Tensor variables negative gradient
    :param variables: float Tensor variables
    :param lower_bounds: float Tensor lower bounds
    :return: bool Tensor indicating if variables converged
    """

    return tf.logical_and(
        tf.less_equal(variables, lower_bounds),
        tf.less_equal(n_grad, tf.zeros_like(n_grad)),
    )


def kuhn_tucker_convergence_test_center(variables, lower_bounds, upper_bounds):
    """
    Convergence test for the variables strictly between the lower and upper bound.

    :param variables: float Tensor variables
    :param lower_bounds: float Tensor lower bounds
    :param upper_bounds: float Tensor upper bounds
    :return: bool Tensor indicating if variables converged
    """

    return tf.logical_and(
        tf.greater(variables, lower_bounds),
        tf.less(variables, upper_bounds),
    )


def kuhn_tucker_convergence_test_upper(n_grad, variables, upper_bounds):
    """
    Convergence test for the variables at the upper bound.
    The gradient for the variables at the upper bound must be positive.

    :param n_grad: float Tensor variables negative gradient
    :param variables: float Tensor variables
    :param upper_bounds: float Tensor upper bounds
    :return: bool Tensor indicating if variables converged
    """

    return tf.logical_and(
        tf.greater_equal(variables, upper_bounds),
        tf.greater_equal(n_grad, tf.zeros(tf.shape(n_grad), dtype=n_grad.dtype)),
    )


def kuhn_tucker_convergence_test(n_grad, variables, lower_bounds, upper_bounds):
    """
    Convergence test for the variables.

    :param n_grad: float Tensor variables negative gradient
    :param variables: float Tensor variables
    :param lower_bounds: float Tensor lower bounds
    :param upper_bounds: float Tensor upper bounds
    :return: bool Tensor indicating if variables converged
    """

    lower_converged = kuhn_tucker_convergence_test_lower(n_grad, variables, lower_bounds)
    center_converged = kuhn_tucker_convergence_test_center(variables, lower_bounds, upper_bounds)
    upper_converged = kuhn_tucker_convergence_test_upper(n_grad, variables, upper_bounds)

    converged = tf.stack([
        lower_converged,
        center_converged,
        upper_converged,
    ], axis=-1)

    return tf.reduce_any(converged, axis=-1)


def free_variable_with_largest_gradient(n_grad, lower_mask, upper_mask):
    """
    Free variable at bound with largest gradient away from the bound.

    :param n_grad: float Tensor variables negative gradient
    :param lower_mask: bool Tensor mask of variables at the lower bound
    :param upper_mask: bool Tensor mask of variables at the upper bound
    :return: (bool Tensor , bool Tensor) lower and upper mask tuple
    """

    lower_values = tf.where(lower_mask, x=n_grad, y=tf.zeros_like(n_grad))
    upper_values = tf.where(upper_mask, x=-n_grad, y=tf.zeros_like(n_grad))
    values = lower_values + upper_values
    v_max = tf.reduce_max(values)
    v_max = tf.maximum(v_max, 1E-9)

    lower_mask = tf.logical_and(
        lower_mask,
        tf.less(lower_values, v_max),
    )

    upper_mask = tf.logical_and(
        upper_mask,
        tf.less(upper_values, v_max),
    )

    return lower_mask, upper_mask


def lstsq_negative_gradient(
        matrix,
        rhs,
        variables,
        axis=1,
        noise_precision=None,
        prior_precision=None,
        target_weights=None):
    if target_weights is None:
        target_weights = tf.ones_like(rhs)

    einsum1 = "ij,j->i" if axis <= 1 else "ijk,ik->ij"
    einsum2 = "ji,j->i" if axis <= 1 else "ikj,ik->ij"

    # Least square
    # matrix = tf.Print(matrix, [tf.shape(matrix)], message="matrix", summarize=10)
    # variables = tf.Print(variables, [tf.shape(variables)], message="variables", summarize=10)
    b = rhs - tf.einsum(einsum1, matrix, variables)
    b = tf.square(target_weights) * b
    w = tf.einsum(einsum2, matrix, b)

    # Prior gradient
    if prior_precision is not None:
        if noise_precision is None:
            noise_precision = tf.constant(1., dtype=matrix.dtype)

        np_sqrt = tf.sqrt(noise_precision)

        return np_sqrt * w + prior_precision * w

    return w


def free_lstsq(
        matrix,
        rhs,
        center_mask,
        lower_mask,
        lower_bounds,
        upper_mask,
        upper_bounds,
        noise_precision=None,
        prior_precision=None,
        target_weights=None,
        l2_regularizer=0.,
        fast=True):
    """
    Least square regression with variables fixed at lower or upper bound values.

    :param matrix: float Tensor design matrix
    :param rhs: float Tensor right hand side
    :param center_mask: bool Tensor mask of free variables
    :param lower_mask: bool Tensor mask of variables at the lower bound
    :param lower_bounds: float Tensor lower bounds
    :param upper_mask: bool Tensor mask of variables at the upper bound
    :param upper_bounds: float Tensor upper bounds
    :param noise_precision: float Scalar noise precision of targets
    :param prior_precision: float Tensor prior precision of variables
    :param target_weights: float Tensor weights of targets
    :param l2_regularizer: float least square regularization
    :param fast: bool fast least square (differentiable but less stable)
    :return: float Tensor least square result for free variables
    """

    if target_weights is None:
        target_weights = tf.ones_like(rhs)

    if prior_precision is not None:
        if noise_precision is None:
            noise_precision = tf.constant(1., dtype=matrix.dtype)

        np_sqrt = tf.sqrt(noise_precision)

        matrix = tf.concat([
            np_sqrt * matrix,
            tf.diag(tf.sqrt(prior_precision)),
        ], axis=0)

        rhs = tf.concat([
            np_sqrt * rhs,
            tf.zeros(tf.shape(prior_precision), dtype=rhs.dtype)
        ], axis=0)

        target_weights = tf.concat([
            target_weights,
            tf.ones(tf.shape(prior_precision), dtype=target_weights.dtype),
        ], axis=0)

    lm = tf.cast(lower_mask, dtype=lower_bounds.dtype)
    um = tf.cast(upper_mask, dtype=upper_bounds.dtype)
    cm = tf.cast(center_mask, dtype=upper_bounds.dtype)

    m = tf.einsum("ij,j->ij", matrix, cm)
    m = tf.einsum("i,ij->ij", target_weights, m)

    b = rhs
    b -= tf.tensordot(matrix, lm * lower_bounds + um * upper_bounds, axes=[[1], [0]])
    b = target_weights * b
    b = tf.expand_dims(b, -1)

    # TODO: performance optimize with QR decomposition
    result = tf.linalg.lstsq(m, b, l2_regularizer=l2_regularizer, fast=fast)
    # result = tf.Print(result, [], message="------------------------", summarize=1000)
    # result = tf.Print(result, [lm], message="BVLS lstsq lm", summarize=1000)
    # result = tf.Print(result, [cm], message="BVLS lstsq cm", summarize=1000)
    # result = tf.Print(result, [um], message="BVLS lstsq um", summarize=1000)
    # result = tf.Print(result, [m], message="BVLS lstsq m", summarize=1000)
    # result = tf.Print(result, [b], message="BVLS lstsq b", summarize=1000)
    # result = tf.Print(result, [result], message="BVLS lstsq", summarize=1000)

    return result[:, 0]


def free_bounded_step(
        center_mask,
        variables,
        lower_mask,
        lower_bounds,
        upper_mask,
        upper_bounds):
    """
    Update variables based on least square result of variables but respecting the bounds.

    :param center_mask: bool Tensor mask of free variables
    :param variables: float Tensor variables
    :param lower_mask: bool Tensor mask of free variables
    :param lower_bounds: float Tensor lower bounds
    :param upper_mask: bool Tensor mask of variables at the upper bound
    :param upper_bounds: float Tensor upper bounds
    :return: (float Tensor, float Tensor): variables and step size
    """

    zero = tf.zeros((), dtype=lower_bounds.dtype)
    one = tf.ones((), dtype=lower_bounds.dtype)

    lm = tf.cast(lower_mask, dtype=lower_bounds.dtype)
    um = tf.cast(upper_mask, dtype=upper_bounds.dtype)
    cm = tf.cast(center_mask, dtype=upper_bounds.dtype)

    lower_alphas = tf.cast(tf.less_equal(lower_bounds, variables), dtype=lower_bounds.dtype)
    upper_alphas = tf.cast(tf.greater_equal(upper_bounds, variables), dtype=lower_bounds.dtype)

    lower_alphas += (1 - lower_alphas) * tf.truediv(lower_bounds, variables)
    upper_alphas += (1 - upper_alphas) * tf.truediv(upper_bounds, variables)

    lower_alphas = (one - cm) + cm * lower_alphas
    upper_alphas = (one - cm) + cm * upper_alphas

    lower_alphas = tf.where(
        tf.less(lower_alphas, zero),
        x=tf.ones_like(lower_alphas),
        y=lower_alphas,
    )
    upper_alphas = tf.where(
        tf.less(upper_alphas, zero),
        x=tf.ones_like(upper_alphas),
        y=upper_alphas,
    )

    min_alpha = tf.reduce_min([lower_alphas, upper_alphas])
    alpha = tf.minimum(one, min_alpha)
    # alpha = tf.Print(alpha, [variables], message="BVLS variables", summarize=1000)
    # alpha = tf.Print(alpha, [lower_alphas], message="BVLS l alpha", summarize=1000)
    # alpha = tf.Print(alpha, [upper_alphas], message="BVLS u alpha", summarize=1000)
    # alpha = tf.Print(alpha, [alpha], message="BVLS alpha", summarize=1000)

    variables = lm * lower_bounds + alpha * cm * variables + um * upper_bounds

    return variables, alpha


def compute_variables_sets(variables, lower_bounds, upper_bounds):
    """
    Compute the lower bound mask, center mask, and upper mask.
    Some numerical error is allowed.

    :param variables: float Tensor variables
    :param lower_bounds: float Tensor lower bounds
    :param upper_bounds: float Tensor upper bounds
    :return: bool Tensor tuple: lower bound variables, free variables, upper bound variables.
    """

    lm = tf.less_equal(variables, lower_bounds + 1E-9)
    um = tf.greater_equal(variables, upper_bounds - 1E-9)
    cm = tf.logical_not(tf.logical_or(lm, um))

    return lm, cm, um


def lstsq_squared_residuals_sum(matrix, variables, rhs, tws):
    residuals = tws * (tf.einsum("ij,j->i", matrix, variables) - rhs)
    return tf.reduce_sum(tf.square(residuals))


# TODO: warm start
# TODO: convergence result
def tf_bvls(
        matrix,
        rhs,
        lower_bounds,
        upper_bounds,
        noise_precision=None,
        prior_precision=None,
        target_weights=None,
        l2_regularizer=0.,
        fast=True,
        maximum_iterations=20,
        return_iterations=False,
        name="bvls",
):
    """
    Least square regression with variable bound

    :param matrix: float Tensor design matrix
    :param rhs: float Tensor right hand side
    :param lower_bounds: float Tensor lower bounds
    :param upper_bounds: float Tensor upper bounds
    :param noise_precision: float Scalar noise precision of targets
    :param prior_precision: float Tensor prior precision of variables
    :param target_weights: float Tensor weights of targets
    :param l2_regularizer: float Scalar regression regularization
    :param fast: bool Use fast regression, less stable
    :param maximum_iterations: int Maximum number of iterations
    :param return_iterations: bool returns number iterations if True
    :param name: str Name of the node in the graph
    :return: float Tensor bounded least square result
    """

    # Validate the inputs
    matrix, rhs, lower_bounds, upper_bounds = tf_bvls_input_validation(
        matrix, rhs, lower_bounds, upper_bounds
    )

    def tf_bvls_condition(_, vs, __, ___, n_grad, free):
        """
        Termination condition

        :param vs: float Tensor variables
        :param n_grad: float Tensor variables negative gradient
        :param free: bool Tensor free variable at boundary with largest gradient
        :return:
        """

        converged = kuhn_tucker_convergence_test(n_grad, vs, lower_bounds, upper_bounds)

        # BVLS terminates when Kuhn-Tucker conditions are met and a variable can be freed
        return tf.logical_not(tf.logical_and(
            free,
            tf.reduce_all(converged),
        ))

    def tf_bvls_body(i, _, lm, um, n_grad, free):
        """
        BVLS least square loop

        :param i: int Scalar iteration counter
        :param lm: bool Tensor variables at lower bound
        :param um: bool Tensor variables at upper bound
        :param n_grad: float Tensor variables negative gradient
        :param free: bool Tensor flag to free variable at boundary with largest gradient
        :return: Tensor tuple with loop variables for next iteration
        """

        lm, um = tf.cond(
            free,
            true_fn=lambda: free_variable_with_largest_gradient(n_grad, lm, um),
            false_fn=lambda: (lm, um),
        )
        cm = tf.logical_not(tf.logical_or(lm, um))

        # Compute the least square regression over the free variables
        result = free_lstsq(
            matrix, rhs, cm, lm, lower_bounds, um, upper_bounds,
            noise_precision=noise_precision,
            prior_precision=prior_precision,
            target_weights=target_weights,
            l2_regularizer=l2_regularizer,
            fast=fast,
        )

        # Perform a bound respecting update step for the variables
        vs, alpha = free_bounded_step(cm, result, lm, lower_bounds, um, upper_bounds)

        # Compute the sets of variables at the lower and upper bounds
        lm, _, um = compute_variables_sets(vs, lower_bounds, upper_bounds)

        # Compute the negative gradient for each variable
        n_grad = lstsq_negative_gradient(
            matrix,
            rhs,
            vs,
            noise_precision=noise_precision,
            prior_precision=prior_precision,
            target_weights=target_weights,
        )

        # When no free variable hit a free bound, the next step is to free
        # the variable at the bound with the largest gradient
        free = tf.greater_equal(alpha, 1.0)

        return i + 1, vs, lm, um, n_grad, free

    # Cold start
    i0 = tf.constant(0, dtype=tf.int8)
    vs0 = (lower_bounds + upper_bounds) / 2.
    lower_mask0 = tf.less_equal(vs0, lower_bounds)
    upper_mask0 = tf.greater_equal(vs0, upper_bounds)
    n_grad0 = lstsq_negative_gradient(matrix, rhs, vs0)
    free0 = tf.constant(False)

    iterations, variables, lower_mask, upper_mask, _, _ = tf.while_loop(
        tf_bvls_condition,
        tf_bvls_body,
        loop_vars=(i0, vs0, lower_mask0, upper_mask0, n_grad0, free0),
        back_prop=False,
        maximum_iterations=maximum_iterations,
        parallel_iterations=1,
        name="bvls_loop",
    )

    if return_iterations:
        return (
            tf.identity(variables, name=name),
            tf.identity(iterations, name="%s_iterations" % name),
        )
    else:
        return tf.identity(variables, name=name)


def tf_bvls_batch(
        matrix,
        rhs,
        lower_bounds,
        upper_bounds,
        noise_precision=None,
        prior_precision=None,
        l2_regularizer=0.,
        fast=True,
        maximum_iterations=20,
        parallel_iterations=None,
        name="bvls_batch"):
    def map_multi_args(fn, arrays, dtype=tf.float32):
        indices = tf.range(0, limit=tf.shape(arrays[0])[0], dtype=tf.int32)
        out = tf.map_fn(
            lambda ii: fn(
                *[array[ii] for array in arrays],
                noise_precision=noise_precision,
                prior_precision=prior_precision,
                l2_regularizer=l2_regularizer,
                fast=fast,
                maximum_iterations=maximum_iterations,
            ),
            indices,
            dtype=dtype,
            parallel_iterations=parallel_iterations,
        )
        return out

    ws = map_multi_args(
        fn=tf_bvls,
        arrays=[matrix, rhs, lower_bounds, upper_bounds],
        dtype=matrix.dtype,
    )

    return tf.identity(ws, name=name)
