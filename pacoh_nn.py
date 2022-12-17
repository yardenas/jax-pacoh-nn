import copy
import functools
from typing import Callable, Iterator, Tuple

import chex
import distrax
import haiku as hk
import jax
import numpy as np
import optax

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
    hyper_posterior = copy.deepcopy(prior)
    keys = hk.PRNGSequence(42)
    for i in range(iterations):
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


@functools.partial(jax.jit, static_argnums=(2, 6, 7))
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
    grad_fn = jax.grad(
        lambda p: particle_mll_loss(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            p,
            hyper_prior,
            key,
            n_prior_samples,
        )
    )
    # vmap to compute the grads for each particle in the ensemble.
    grads = jax.vmap(grad_fn)(hyper_posterior.params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(prior.params, updates)
    return models.ParamsMeanField(new_params), new_opt_state  # type: ignore


def particle_mll_loss(
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


# # Based on tf-probability implementation of batched pairwise matrices:
# # https://github.com/tensorflow/probability/blob
# # /f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python
# # /math/psd_kernels/internal/util.py#L190
# def rbf_kernel(x, y, bandwidth=None):
#     row_norm_x = (x**2).sum(-1)[..., None]
#     row_norm_y = (y**2).sum(-1)[..., None, :]
#     pairwise = jnp.clip(row_norm_x + row_norm_y - 2.0 * jnp.matmul(x, y.T), 0.0)
#     n_x = pairwise.shape[-2]
#     bandwidth = bandwidth or jnp.median(pairwise)
#     bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
#     bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
#     k_xy = jnp.exp(-pairwise / bandwidth / 2)
#     return k_xy


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
