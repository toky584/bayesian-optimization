import jax.numpy as jnp
from forrester import forrester_funct

def acquisition_funct(mu, variance, kappa=4.0):
    """
    Acquisition function: Lower Confidence Bound.
    """
    return mu - kappa * variance

def least_confident(X_train, X_test, y_train, acquisition_funct, mean_preds, std_preds):
    """
    Finds the least confident prediction (minimum of the acquisition function)
    and moves it from X_test to X_train.
    """
    # Compute the acquisition function across the test predictions
    acqu = acquisition_funct(mean_preds, std_preds**2)
    index_max = jnp.argmin(acqu)
    
    # Identify the new point to evaluate
    x_new = X_test[index_max]
    
    # Append the new point to X_train
    if X_train.ndim > 1:
        X_train_new = jnp.vstack([X_train, x_new])
    else:
        X_train_new = jnp.append(X_train, x_new)
        
    # Evaluate the objective function (Forrester function) at the new point
    x_val = x_new[0] if x_new.ndim > 0 else x_new
    y_new = forrester_funct(x_val)
    
    # Append the evaluated point to y_train
    y_train_new = jnp.append(y_train, y_new)
    
    # Remove the selected point from X_test
    X_test_new = jnp.delete(X_test, index_max, axis=0)
    
    return X_train_new, X_test_new, y_train_new
