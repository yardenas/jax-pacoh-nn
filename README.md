# Implementation

## Training
1. Sample data from meta batch
2. Compute PACOH log probs and grads.
3. Compute repulsion force a la. https://github.com/ratschlab/repulsive_ensembles/blob/master/methods/WGD.py
4. Apply grads on meta-posterior parameters.

## How to compute PACOH?
1. Use the hyper posterior particles to sample a new prior particles.
2. Feed forward all tasks for all prior samples.
3. Apply logsumexp and normalize.
4. Scale as needed.
5. Compute the log likelihood of the hyper-prior.
6. Take gradient of the MAP w.r.t. hyper-posterior particles.