import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS
from gp import gp_model, gp_predict

def mcmc_runing(X_train, X_test, y_train, rng_key=None, num_warmup=1500, num_samples=500, num_chains=4):
    """
    Run MCMC using NUTS to sample from the posterior of the GP.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
    nuts_kernel = NUTS(gp_model)
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains, 
        progress_bar=True
    )
    mcmc.run(rng_key, X_train, y_train)
    posterior_samples = mcmc.get_samples()
    
    mean_preds, std_preds = gp_predict(X_train, X_test, y_train, posterior_samples)
    return mean_preds, std_preds
