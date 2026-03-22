import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def matern_3_2_kernel(X1, X2, length_scale=1.0, variance=1.0):
    # Compute pairwise distances
    dist_matrix = jnp.abs(X1 - X2.T)  # Pairwise absolute distance
    factor = jnp.sqrt(3.0) * dist_matrix / length_scale
    return variance * (1.0 + factor) * jnp.exp(-factor)

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

def gp_predict(X_train, X_test, y_train, posterior_samples):
    mean_preds = []
    var_preds = []
    
    for length_scale, variance, noise in zip(
        posterior_samples["length_scale"],
        posterior_samples["variance"],
        posterior_samples["noise"],
    ):
        # Kernel for training-test points and test-test points
        K_train_test = matern_3_2_kernel(X_train, X_test, length_scale=length_scale, variance=variance)
        K_test_test = matern_3_2_kernel(X_test, X_test, length_scale=length_scale, variance=variance)

        # Kernel for training data
        K_train = matern_3_2_kernel(X_train, X_train, length_scale=length_scale, variance=variance)
        
        # Compute the conditional mean and covariance for predictions
        K_inv = jnp.linalg.inv(K_train + noise * jnp.eye(X_train.shape[0]))
        mean = jnp.dot(K_train_test.T, jnp.dot(K_inv, y_train))
        cov = K_test_test - jnp.dot(K_train_test.T, jnp.dot(K_inv, K_train_test))

        mean_preds.append(mean)
        var_preds.append(jnp.diag(cov))

    # Averaging posterior predictive samples
    mean_preds = jnp.mean(jnp.array(mean_preds), axis=0)
    var_preds = jnp.mean(jnp.array(var_preds), axis=0)
    std_preds = jnp.sqrt(var_preds)

    return mean_preds, std_preds
