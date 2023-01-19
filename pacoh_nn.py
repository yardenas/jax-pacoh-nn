import copy
import functools
from typing import Any, Callable, Iterator, Optional, Tuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree  # type: ignore

import models


def meta_train(
    data: Iterator[Tuple[np.ndarray, np.ndarray]],
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
    hyper_prior: models.ParamsMeanField,
    hyper_posterior: models.ParamsMeanField,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    iterations: int,
    n_prior_samples: int,
) -> models.ParamsMeanField:
    """Approximate inference of a hyper-posterior, given a hyper-prior and prior.

    Args:
        data (Iterator[Tuple[np.ndarray, np.ndarray]]): The dataset to be learned.
        prediction_fn (Callable[[chex.ArrayTree, chex.Array], chex.Array]): Parameterizd
        function approximator.
        hyper_prior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        hyper_posterior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        optimizer (optax.GradientTransformation): Optimizer.
        opt_state (optax.OptState): Optimizer state.
        iterations (int): Number of update iterations to be performed
        n_prior_samples (int): Number of prior samples to draw for each task.

    Returns:
        models.ParamsMeanField: Trained hyper-posterior.
    """
    hyper_posterior = copy.deepcopy(hyper_posterior)
    keys = hk.PRNGSequence(42)
    for i in range(iterations):
        meta_batch_x, meta_batch_y = next(data)
        hyper_posterior, opt_state, log_probs = train_step(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            hyper_prior,
            hyper_posterior,
            next(keys),
            n_prior_samples,
            optimizer,
            opt_state,
        )
        if i % 100 == 0:
            print(f"Iteration {i} log probs: {log_probs}")
    return hyper_posterior


@functools.partial(jax.jit, static_argnums=(2, 6, 7))
def train_step(
    meta_batch_x: chex.Array,
    meta_batch_y: chex.Array,
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
    hyper_prior: models.ParamsMeanField,
    hyper_posterior: models.ParamsMeanField,
    key: chex.PRNGKey,
    n_prior_samples: int,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> Tuple[models.ParamsMeanField, optax.OptState, jnp.ndarray]:
    """Approximate inference of a hyper-posterior, given a hyper-prior and prior.

    Args:
        meta_batch_x: Meta-batch of input data.
        meta_batch_y: Meta-batch of output data.
        prediction_fn: Parameterized function approximator.
        hyper_prior: Prior distribution over distributions of parameterized functions.
        hyper_posterior: Infered posterior distribution over distributions
            parameterized functions.
        key: PRNG key for stochasticity.
        n_prior_samples: Number of prior samples to draw for each task.
        optimizer: Optimizer.
        opt_state: Initial optimizer state.

    Returns:
        Trained hyper-posterior and optimizer state.
    """
    grad_fn = jax.value_and_grad(
        lambda hyper_posterior: particle_loss(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            hyper_posterior,
            hyper_prior,
            key,
            n_prior_samples,
        )
    )
    # vmap to compute the grads for each particle in the ensemble with respect
    # to its prediction's log probability.
    log_probs, log_prob_grads = jax.vmap(grad_fn)(hyper_posterior)
    # Compute the particles' kernel matrix and its per-particle gradients.
    num_particles = jax.tree_util.tree_flatten(log_prob_grads)[0][0].shape[0]
    particles_matrix, reconstruct_tree = _to_matrix(hyper_posterior, num_particles)
    kxx, kernel_vjp = jax.vjp(
        lambda x: rbf_kernel(x, particles_matrix), particles_matrix
    )
    # Summing along the 'particles axis' to compute the per-particle gradients, see
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    kernel_grads = kernel_vjp(jnp.ones(kxx.shape))[0]
    stein_grads = (
        jnp.matmul(kxx, _to_matrix(log_prob_grads, num_particles)[0]) + kernel_grads
    ) / num_particles
    stein_grads = reconstruct_tree(stein_grads.ravel())
    updates, new_opt_state = optimizer.update(stein_grads, opt_state)
    new_params = optax.apply_updates(hyper_posterior, updates)
    return (models.ParamsMeanField(*new_params), new_opt_state, log_probs.mean())


def _to_matrix(
    params: chex.ArrayTree, num_particles: int
) -> Tuple[chex.Array, Callable[[chex.Array], hk.Params]]:
    flattened_params, reconstruct_tree = ravel_pytree(params)
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


def particle_loss(
    meta_batch_x: chex.Array,
    meta_batch_y: chex.Array,
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
    particle: models.ParamsMeanField,
    hyper_prior: models.ParamsMeanField,
    key: chex.PRNGKey,
    n_prior_samples: int,
) -> chex.Array:
    """Computes the loss of each SVGD particle of PACOH
    (l. 7, Algorithm 1 PACOH with SVGD approximation of Q*).

    Args:
        meta_batch_x (Array): Input array [meta_batch_dim, batch_dim, input_dim]
        meta_batch_y (Array): Output array [meta_batch_dim, batch_dim, output_dim]
        prediction_fn (Callable[[hk.Params, Array], Array]): Parameterized function
        approximator.
        params (hk.Params): Particle's parameters to learn
        key (PRNGKey): Key for stochasticity.
        n_prior_samples (int): Number of samples.
    Returns:
        Array: Loss.
    """

    def estimate_mll(x: chex.Array, y: chex.Array) -> chex.Array:
        prior_samples = particle.sample(key, n_prior_samples)
        per_sample_pred = jax.vmap(prediction_fn, (0, None))
        y_hat, stddevs = per_sample_pred(prior_samples, x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(y)
        batch_size = x.shape[0]
        mll = jax.scipy.special.logsumexp(
            log_likelihood, axis=0, b=jnp.sqrt(batch_size)
        ) - np.log(n_prior_samples)
        return mll

    # vmap estimate_mll over the task batch dimension, as specified
    # @ Algorithm 1 PACOH with SVGD approximation of Q∗ (MLL_Estimator)
    # @ https://arxiv.org/pdf/2002.05551.pdf.
    mll = jax.vmap(estimate_mll)(meta_batch_x, meta_batch_y)
    log_prob_prior = hyper_prior.log_prob(particle)
    return -(mll + log_prob_prior).mean()


# Based on tf-probability implementation of batched pairwise matrices:
# https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/math/psd_kernels/internal/util.py#L190
def rbf_kernel(
    x: chex.Array, y: chex.Array, bandwidth: Optional[chex.Numeric] = None
) -> chex.Array:
    """Computes the RBF kernel matrix between (batches of) x and y.
    Returns (batches of) kernel matrices
    :math:`K(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))`.
    """
    row_norm_x = (x**2).sum(-1)[..., None]
    row_norm_y = (y**2).sum(-1)[..., None, :]
    pairwise = jnp.clip(row_norm_x + row_norm_y - 2.0 * jnp.matmul(x, y.T), 0.0)
    n_x = pairwise.shape[-2]
    bandwidth = bandwidth if bandwidth is not None else jnp.median(pairwise)
    bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
    bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
    k_xy = jnp.exp(-pairwise / bandwidth / 2)
    return k_xy


@functools.partial(jax.jit, static_argnums=(3, 5))
def infer_posterior(
    x: chex.Array,
    y: chex.Array,
    hyper_posterior: models.ParamsMeanField,
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
    key: chex.PRNGKey,
    update_steps: int,
    learning_rate: float,
) -> Tuple[chex.ArrayTree, chex.Array]:
    """Infer posterior based on task specific training data.
    The posterior is modeled as an ensemble of neural networks.

    Args:
        x (chex.Array): x-values of task-specific training data.
        [task_dim, batch_dim, input_dim]
        y (chex.Array): y-values of task-specific training data.
        [task_dim, batch_dim, output_dim]
        hyper_posterior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        prediction_fn (Callable[[chex.ArrayTree, chex.Array], chex.Array]):
        parameterizd function.
        key (chex.PRNGKey): PRNG key.
        update_steps (int): Number of update steps to be performed.

    Returns:
        models.ParamsMeanField: Task-inferred posterior.
    """
    # Sample prior parameters from the hyper-posterior to form an ensemble
    # of neural networks.
    posterior_params = hyper_posterior.sample(key, 1)
    posterior_params = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), posterior_params
    )
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(posterior_params)

    def loss(params: hk.Params) -> chex.Array:
        y_hat, stddevs = prediction_fn(params, x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(y)
        return -log_likelihood.mean()

    def update(
        carry: Tuple[chex.ArrayTree, optax.OptState], _: Any
    ) -> Tuple[Tuple[chex.ArrayTree, optax.OptState], chex.Array]:
        posterior_params, opt_state = carry
        values, grads = jax.vmap(jax.value_and_grad(loss))(posterior_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        posterior_params = optax.apply_updates(posterior_params, updates)
        return (posterior_params, opt_state), values.mean()

    (posterior_params, _), losses = jax.lax.scan(
        update, (posterior_params, opt_state), None, update_steps
    )
    return posterior_params, losses


@functools.partial(jax.jit, static_argnums=(2))
def predict(
    posterior: chex.ArrayTree,
    x: chex.Array,
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
) -> Tuple[chex.Array, chex.Array]:
    """Predict y-values based on the posterior (defined by an ensemble of
     neural networks).
    Args:
        posterior (chex.ArrayTree): Posterior parameters.
        x (chex.Array): x-values of task-specific training data.
        prediction_fn (Callable[[chex.ArrayTree, chex.Array], chex.Array]): Parameterized
        function.

    Returns:
        chex.Array: Prediced mean and standard deviation predicted by each member
        of the ensemble that defines the ensemble.
    """
    prediction_fn = jax.vmap(prediction_fn, in_axes=(0, None))
    y_hat, stddev = prediction_fn(posterior, x)
    return y_hat, stddev


if __name__ == "__main__":
    import functools

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    import models
    import pacoh_nn as pacoh
    import sinusoid_regression_dataset

    dataset = sinusoid_regression_dataset.SinusoidRegression(16, 5, 5)

    def net(x: jnp.ndarray) -> jnp.ndarray:
        x = hk.nets.MLP((32, 32, 32, 32, 1))(x)
        mu = x
        # mu, stddev = jnp.split(x, 2, -1)
        # stddev = jnp.clip(jnn.softplus(stddev), 1e-3, 0.5)
        stddev = hk.get_parameter(
            "stddev", [], init=lambda shape, dtype: jnp.ones(shape, dtype) * 1e-3
        )
        return mu, stddev * jnp.ones_like(mu)

    init, apply = hk.without_apply_rng(hk.transform(net))
    example = next(dataset.train_set)[0][0]
    seed_sequence = hk.PRNGSequence(666)
    mean_prior_over_mus = jax.tree_map(
        jnp.zeros_like, init(next(seed_sequence), example)
    )
    mean_prior_over_stddevs = jax.tree_map(jnp.zeros_like, mean_prior_over_mus)
    hyper_prior = models.ParamsMeanField(
        models.ParamsMeanField(mean_prior_over_mus, mean_prior_over_stddevs),
        1.0,
    )
    n_particles = 10
    init = jax.vmap(init, (0, None))
    particles_mus = init(jnp.asarray(seed_sequence.take(n_particles)), example)
    particle_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * 1e-4, particles_mus)
    hyper_posterior = models.ParamsMeanField(particles_mus, particle_stddevs)
    opt = optax.flatten(optax.adam(3e-4))
    opt_state = opt.init(hyper_posterior)
    hyper_posterior = pacoh.meta_train(
        dataset.train_set, apply, hyper_prior, hyper_posterior, opt, opt_state, 1000, 10
    )
    easy_dataset = sinusoid_regression_dataset.SinusoidRegression(16, 5, 100)
    infer_posteriors = jax.vmap(
        pacoh.infer_posterior, in_axes=(0, 0, None, None, None, None, None)
    )

    (context_x, context_y), (test_x, test_y) = next(easy_dataset.test_set)
    posteriors, losses = infer_posteriors(
        context_x, context_y, hyper_posterior, apply, next(seed_sequence), 1000, 3e-4
    )
    predict = jax.vmap(predict, (0, 0, None))
    predictions = predict(posteriors, test_x, apply)
