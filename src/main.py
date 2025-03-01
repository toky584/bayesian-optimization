#Main loop.
def loop_sampling(X_train, X_test, y_train, number_evaluation):
    mean_preds, std_preds = mcmc_runing(gp_model, X_train, X_test, y_train)
    while len(X_train.tolist()) < number_evaluation:
        X_train, X_test, y_train = least_confident(X_train, X_test, y_train, acquisition_funct, mean_preds, std_preds)
        mean_preds, std_preds = mcmc_runing(gp_model, X_train, X_test, y_train)
    return X_train, y_train, X_test, mean_preds, std_preds
