#compute kernel
def matern_3_2_kernel(X1, X2, length_scale=1.0, variance=1.0):
    # Compute pairwise distances
    dist_matrix = jnp.abs(X1 - X2.T)  # Pairwise absolute distance
    factor = jnp.sqrt(3) * dist_matrix / length_scale
    return variance * (1 + factor) * jnp.exp(-factor)

# GP model definition
def gp_model(X_train, y_train):
    # Priors for kernel parameters
    length_scale = numpyro.sample("length_scale", dist.LogNormal(0.0, 1.0))
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))

    # Kernel
    K = matern_3_2_kernel(X_train, X_train, length_scale=length_scale, variance=variance)

    # Prior for noise
    noise = numpyro.sample("noise", dist.LogNormal(-1.0, 0.5))

    # Adding noise to the diagonal of the kernel matrix (for observations)
    K_noise = K + noise * jnp.eye(X_train.shape[0])

    # GP definition (using MultivariateNormal as the prior)
    f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(X_train.shape[0]), covariance_matrix=K_noise))

    # Observation likelihood
    numpyro.sample("y", dist.Normal(f, noise), obs=y_train)
