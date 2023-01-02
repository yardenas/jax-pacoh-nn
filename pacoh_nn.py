import copy
from typing import Callable, Iterator, Optional, Tuple

import chex
import distrax
import haiku as hk
import jax
import numpy as np
import optax
from jax.flatten_util import ravel_pytree  # type: ignore

import models


def meta_train(
    data: Iterator[Tuple[np.ndarray, np.ndarray]],
    prediction_fn: Callable[[hk.Params, chex.Array], chex.Array],
    hyper_prior: models.ParamsMeanField,
    prior: models.ParamsMeanField,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    iterations: int,
    n_prior_samples: int,
) -> models.ParamsMeanField:
    """Approximate inference of a hyper-posterior, given a hyper-prior and prior.

    Args:
        data (Iterator[Tuple[np.ndarray, np.ndarray]]): The dataset to be learned.
        prediction_fn (Callable[[hk.Params, chex.Array], chex.Array]): Parameterizd
        function approximator.
        hyper_prior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        prior (models.ParamsMeanField): Distribution over parameterized functions.
        optimizer (optax.GradientTransformation): Optimizer.
        opt_state (optax.OptState): Optimizer state.
        iterations (int): Number of update iterations to be performed
        n_prior_samples (int): Number of prior samples to draw for each task.

    Returns:
        models.ParamsMeanField: Trained hyper-posterior.
    """
    hyper_posterior = copy.deepcopy(prior)
    keys = hk.PRNGSequence(42)
    for _ in range(iterations):
        meta_batch_x, meta_batch_y = next(data)
        hyper_posterior, opt_state = train_step(
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
    return hyper_posterior


# @functools.partial(jax.jit, static_argnums=(2, 6, 7))
def train_step(
    meta_batch_x: chex.Array,
    meta_batch_y: chex.Array,
    prediction_fn: Callable[[hk.Params, chex.Array], chex.Array],
    hyper_prior: models.ParamsMeanField,
    hyper_posterior: models.ParamsMeanField,
    key: chex.PRNGKey,
    n_prior_samples: int,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> Tuple[models.ParamsMeanField, optax.OptState]:
    """Approximate inference of a hyper-posterior, given a hyper-prior and prior.

    Args:
        meta_batch_x (chex.Array): Meta-batch of input data.
        meta_batch_y (chex.Array): Meta-batch of output data.
        prediction_fn (Callable[[hk.Params, chex.Array], chex.Array]): Parameterized
        function approximator.
        hyper_prior (models.ParamsMeanField): Prior distribution over distributions of
        parameterized functions.
        hyper_posterior (models.ParamsMeanField): Infered posterior distribution over
        distributions parameterized functions.
        key (chex.PRNGKey): PRNG key for stochasticity.
        n_prior_samples (int): Number of prior samples to draw for each task.
        optimizer (optax.GradientTransformation): Optimizer.
        opt_state (optax.OptState): Initial optimizer state.

    Returns:
        Tuple[models.ParamsMeanField, optax.OptState]:
        Trained hyper-posterior and optimizer state.
    """
    grad_fn = jax.grad(
        lambda p: particle_loss(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            p,
            hyper_prior,
            key,
            n_prior_samples,
        )
    )
    # vmap to compute the grads for each particle in the ensemble with respect
    # to its prediction's log probability.
    log_prob_grads = jax.vmap(grad_fn)(hyper_posterior.params)
    # Compute the particles' kernel matrix and its per-particle gradients.
    num_particles = jax.tree_util.tree_flatten(log_prob_grads)[0][0].shape[0]
    particles_matrix, reconstruct_tree = _to_matrix(
        hyper_posterior.params, num_particles
    )
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
    new_params = optax.apply_updates(prior.params, updates)
    return models.ParamsMeanField(new_params), new_opt_state  # type: ignore


def _to_matrix(
    params: hk.Params, num_particles: int
) -> Tuple[chex.Array, Callable[[chex.Array], hk.Params]]:
    flattened_params, reconstruct_tree = ravel_pytree(params)
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


def particle_loss(
    meta_batch_x: chex.Array,
    meta_batch_y: chex.Array,
    prediction_fn: Callable[[hk.Params, chex.Array], chex.Array],
    params: hk.Params,
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
        prior_samples = models.ParamsMeanField(params).sample(key, n_prior_samples)
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
    log_prob_prior = hyper_prior.log_prob(params)
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


if __name__ == "__main__":
    import haiku as hk
    import jax.numpy as jnp
    import optax

    import models
    import sinusoid_regression_dataset

    dataset = sinusoid_regression_dataset.SinusoidRegression(16, 5, 666)

    def net(x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x = hk.nets.MLP((32, 32, 32, 32, 2))(x)
        mu, stddev = jnp.split(x, 2, -1)
        return mu, stddev

    init, apply = hk.without_apply_rng(hk.transform(net))
    example = next(dataset.train_set)[0][0]
    seed_sequence = hk.PRNGSequence(666)
    hyper_prior = models.ParamsMeanField(
        jax.tree_map(jnp.zeros_like, init(next(seed_sequence), example))
    )
    n_particles = 10
    init = jax.vmap(init, (0, None))
    prior_particles = init(jnp.asarray(seed_sequence.take(n_particles)), example)
    prior = models.ParamsMeanField(prior_particles)
    opt = optax.flatten(optax.adam(2e-3))
    opt_state = opt.init(prior_particles)
    meta_train(dataset.train_set, apply, hyper_prior, prior, opt, opt_state, 1, 10)
