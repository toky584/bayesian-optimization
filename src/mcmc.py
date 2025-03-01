# function for predict using posterior samples
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
        K_train_test_T = K_train_test.T
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

# Mcmc running
def mcmc_runing(model, X_train, X_test, y_train):   
    nuts_kernel = NUTS(gp_model)
    mcmc = MCMC(nuts_kernel, num_warmup=1500, num_chains = 4, num_samples=500, progress_bar = True)
    mcmc.run(rng_key, X_train, y_train)
    posterior_samples = mcmc.get_samples()
    mean_preds, std_preds = gp_predict(X_train, X_test, y_train, posterior_samples)
    return mean_preds, std_preds
