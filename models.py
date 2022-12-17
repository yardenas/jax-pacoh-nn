from typing import Any, Callable, NamedTuple, Tuple

import chex
import distrax
import haiku as hk
import jax
from jax.flatten_util import ravel_pytree  # type: ignore


class ParamsMeanField(NamedTuple):
    params: hk.Params

    def log_prob(self, params: hk.Params) -> chex.Array:
        """Measures the log probability of (batches of) parameter samples.

        Args:
            params (hk.Params): Parameters of a model.

        Returns:
            Array: Log probability of each params sample.
        """
        dist, self_flat_params, _ = self._to_dist()
        flat_params, _ = ravel_pytree(params)
        if len(flat_params) != len(self_flat_params):
            quotient, remainder = divmod(len(flat_params), len(self_flat_params))
            assert (
                remainder == 0
            ), "Given parameters are not given in the form of batches of parameters."
            flat_params = flat_params.reshape((quotient, len(self_flat_params)))
        return dist.log_prob(flat_params)

    def sample(self, seed: chex.PRNGKey, n_samples: int) -> hk.Params:
        """Samples parameters.

        Args:
            seed (PRNGKey): Seed needed for stateless sampling.
            n_samples (int): The number of params samples to return.

        Returns:
            Array: N samples of params, vectorized to the first axis.
        """
        dist, _, pytree_def = self._to_dist()
        samples = dist.sample(seed=seed, sample_shape=(n_samples,))
        pytree_def = jax.vmap(pytree_def)
        return pytree_def(samples)

    def _to_dist(
        self,
    ) -> Tuple[distrax.Distribution, chex.Array, Callable[[chex.Array], Any]]:
        self_flat_params, pytree_def = ravel_pytree(self.params)
        dist = distrax.MultivariateNormalDiag(self_flat_params)
        return dist, self_flat_params, pytree_def
