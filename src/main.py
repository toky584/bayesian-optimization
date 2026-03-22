import jax.numpy as jnp
from mcmc import mcmc_runing
from acquisition import least_confident, acquisition_funct
from forrester import forrester_funct

def loop_sampling(X_train, X_test, y_train, number_evaluation):
    """
    Iteratively sample points using the acquisition function and MCMC to update the GP.
    """
    mean_preds, std_preds = mcmc_runing(X_train, X_test, y_train)
    
    while len(X_train) < number_evaluation:
        print(f"Current evaluations: {len(X_train)}. Evaluating next point...")
        X_train, X_test, y_train = least_confident(
            X_train, X_test, y_train, acquisition_funct, mean_preds, std_preds
        )
        mean_preds, std_preds = mcmc_runing(X_train, X_test, y_train)
        
    return X_train, y_train, X_test, mean_preds, std_preds

if __name__ == "__main__":
    # Example initialization
    # Start with 3 initial points
    X_train = jnp.array([[0.0], [0.5], [1.0]])
    y_train = jnp.array([forrester_funct(x[0]) for x in X_train])
    
    # Test set to query from
    X_test = jnp.linspace(0.0, 1.0, 100).reshape(-1, 1)
    
    print("Starting Bayesian Optimization...")
    X_train_final, y_train_final, X_test_final, mean_preds, std_preds = loop_sampling(
        X_train, X_test, y_train, number_evaluation=6
    )
    
    print(f"\nFinal evaluations completed: {len(X_train_final)}")
    print(f"Final sampled points (X):\n{X_train_final}")
    print(f"Final sampled values (y):\n{y_train_final}")
