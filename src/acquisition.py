def acquisition_funct(mu, variance, kappa = 4):
    return mu - kappa*variance
def least_confident(X_train, X_test, y_train, acquisition_funct, mean_preds, std_preds):
    list_acqu = []
    for i in range(len(mean_preds)):
        acqui = acquisition_funct(mean_preds[i], std_preds[i]**2)
        list_acqu.append(acqui)
    index_max = jnp.argmin(jnp.array(list_acqu))
    X_train_new = X_train.tolist()
    X_train_new.append(X_test[index_max].tolist())
    X_train_new = jnp.array(X_train_new)
    y_train_new = y_train.tolist()
    y_train_new.append(forrester_funct(X_test[index_max][0]))
    y_train_new = jnp.array(y_train_new)
    X_test_new = X_test.tolist()
    X_test_new.remove(X_test[index_max].tolist())
    X_test_new = jnp.array(X_test_new)
    return X_train_new, X_test_new, y_train_new
