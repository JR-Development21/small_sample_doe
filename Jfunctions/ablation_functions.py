import numpy as np
import pandas as pd
import itertools

from Jfunctions import util_functions

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import balanced_accuracy_score, log_loss, brier_score_loss, make_scorer
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from skopt import BayesSearchCV
from skopt.space import Real, Integer



neg_brier_scorer = make_scorer(
    brier_score_loss,
    greater_is_better=False,
    response_method="predict_proba",
    pos_label='P' # Optionally specify the positive label
)

def ablation_analysis(
        dataLoader,
        modelLoader,
        designLoader,
        gridSize,
        seed
):
    Xtrain = dataLoader["train_data"].iloc[:, dataLoader["feature_cols"]]
    ytrain = dataLoader["train_data"].iloc[:, dataLoader["target_col"]]

    Xtest = dataLoader["test_data"].iloc[:, dataLoader["feature_cols"]]
    ytest = dataLoader["test_data"].iloc[:, dataLoader["target_col"]]

    hyperparameters = modelLoader["hyperparameters"]
    n_hyp = len(hyperparameters)

    rep_hyp = int(np.ceil(gridSize ** (1 / n_hyp)))

    logList = modelLoader["logList"]
    if logList is None:
        logList = [False] * n_hyp

    lows = modelLoader["hyperparameter_low"]
    highs = modelLoader["hyperparameter_high"]
    round_param = modelLoader["round_param"]
    if round_param is None:
        round_param = [False] * n_hyp

    base_model = modelLoader["model_constructor"]
    param_grid_dict = {}

    for i, hp in enumerate(hyperparameters):
        if logList[i]:
            param_grid_dict[f"model__{hp}"] = np.logspace(
                np.log10(lows[i]),
                np.log10(highs[i]),
                rep_hyp
            )
        else:
            param_grid_dict[f"model__{hp}"] = np.linspace(lows[i], highs[i], rep_hyp)

        if round_param[i]:
                    param_grid_dict[f"model__{hp}"] = np.unique(np.round(param_grid_dict[f"model__{hp}"]).astype(int))

    cartesian_tuples = itertools.product(*param_grid_dict.values())
    result_array = np.array(list(cartesian_tuples))

    param_grid_cart_dict = {}

    for i, hp in enumerate(hyperparameters):
        param_grid_cart_dict[f"model__{hp}"] = result_array[:,i]
        if round_param[i]:
            param_grid_cart_dict[f"model__{hp}"] = param_grid_cart_dict[f"model__{hp}"].astype(int)

    des_grid_real = pd.DataFrame()
    params_space = {}
    
    for i, hp in enumerate(hyperparameters):
        des_grid_real[hp] = util_functions.design_log_rescaler(
            designLoader.iloc[:, i].values,
            lows[i],
            highs[i],
            logList[i]
        )
        val = des_grid_real[hp]
        
        if modelLoader["round_param"][i]:
            params_space[f"model__{hp}"] = np.round(val).astype(int)
        else:
            params_space[f"model__{hp}"] = val

    custom_configs = [
        dict(zip(params_space.keys(), values))
        for values in zip(*params_space.values())
    ]

    param_grid_space = [
        {key: [value] for key, value in config.items()}
        for config in custom_configs
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # param grid is for CV
    # param grid cart is for RSM
    # param space is for RSM
    # param grid space is for CV

    space_ba_emp_test, space_ba_emp_val = empirical_min(Xtrain, ytrain, Xtest, ytest, base_model,param_grid_space,True,cv) 
    space_mse_emp_test, space_mse_emp_val = empirical_min(Xtrain, ytrain, Xtest, ytest, base_model,param_grid_space,False,cv) 
    space_mse_rsm_2_test, space_mse_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,params_space, hyperparameters, n_hyp, round_param, logList, lows, highs, False,False,cv) 
    space_ba_rsm_2_test, space_ba_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,params_space, hyperparameters, n_hyp, round_param, logList, lows, highs, True,False,cv) 
    grid_mse_rsm_2_test, grid_mse_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,param_grid_cart_dict, hyperparameters, n_hyp, round_param, logList, lows, highs, False,False,cv) 
    grid_ba_rsm_2_test, grid_ba_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,param_grid_cart_dict, hyperparameters, n_hyp, round_param, logList, lows, highs, True,False,cv) 
    grid_mse_rsm_3_test, grid_mse_rsm_3_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,param_grid_cart_dict, hyperparameters, n_hyp, round_param, logList, lows, highs, False,True,cv) 
    grid_ba_rsm_3_test, grid_ba_rsm_3_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,param_grid_cart_dict, hyperparameters, n_hyp, round_param, logList, lows, highs, True,True,cv)

    return {
         "space_ba_emp_test": space_ba_emp_test,
         "space_ba_emp_val": space_ba_emp_val,
         "space_mse_emp_test": space_mse_emp_test,
         "space_mse_emp_val": space_mse_emp_val,
         "space_mse_rsm_2_test": space_mse_rsm_2_test,
         "space_mse_rsm_2_val": space_mse_rsm_2_val,
         "space_ba_rsm_2_test": space_ba_rsm_2_test,
         "space_ba_rsm_2_val": space_ba_rsm_2_val,
         "grid_mse_rsm_2_test": grid_mse_rsm_2_test,
         "grid_mse_rsm_2_val": grid_mse_rsm_2_val,
         "grid_ba_rsm_2_test": grid_ba_rsm_2_test,
         "grid_ba_rsm_2_val": grid_ba_rsm_2_val,
         "grid_mse_rsm_3_test": grid_mse_rsm_3_test,
         "grid_mse_rsm_3_val": grid_mse_rsm_3_val,
         "grid_ba_rsm_3_test": grid_ba_rsm_3_test,
         "grid_ba_rsm_3_val": grid_ba_rsm_3_val,
    }


def empirical_min(
          Xtrain,
          ytrain,
          Xtest,
          ytest,
          base_model,
          param_grid,
          ba_bool,
          cv
):
    search = GridSearchCV(
                base_model(),
                param_grid=param_grid,
                scoring="balanced_accuracy" if ba_bool else neg_brier_scorer,
                cv=cv,
                n_jobs=1
            )
    search.fit(Xtrain, ytrain)
    
    best_model = search.best_estimator_

    yhat_test = best_model.predict(Xtest)

    grid_search_test_acc = balanced_accuracy_score(ytest, yhat_test)
    grid_search_val_acc = cross_val_score(
                                        best_model,
                                        Xtrain,
                                        ytrain,
                                        scoring="balanced_accuracy",
                                        cv=cv,
                                        n_jobs=1
                                    ).mean()
    return grid_search_test_acc, grid_search_val_acc

def expected_min(
          Xtrain,
          ytrain,
          Xtest,
          ytest,
          base_model,
          param_grid,
          hyperparameters,
          n_hyp,
          round_param,
          logList,
          lows,
          highs,
          ba_bool,
          cubic_bool,
          cv
):
    des_grid_real = pd.DataFrame(param_grid)
    des_scores = []

    for _, row in des_grid_real.iterrows():
        params = {}
        for i, hp in enumerate(hyperparameters):
            params[hp] = row[f"model__{hp}"]
            if round_param[i]:
                params[hp] = np.round(params[hp]).astype(int)


        model = base_model(**params)

        scores = cross_val_score(
            model,
            Xtrain,
            ytrain,
            scoring="balanced_accuracy" if ba_bool else neg_brier_scorer,
            cv=cv
        )
        des_scores.append(-scores.mean())

    df_surrogate = pd.DataFrame()

    for i, hp in enumerate(hyperparameters):
        df_surrogate[hp] = util_functions.design_log_scaler(
            des_grid_real[f"model__{hp}"].values,
            lows[i],
            highs[i],
            logList[i]
        )
    

    df_surrogate["Accuracy"] = des_scores

    terms = hyperparameters

    main_terms = " + ".join(terms)
    quad_terms = " + ".join([f"I({t}**2)" for t in terms])


    interaction_terms = " + ".join(
        f"{t1}:{t2}"
        for i, t1 in enumerate(terms)
        for t2 in terms[i+1:]
    )
    if cubic_bool:
        cubic_terms = " + ".join([f"I({t}**3)" for t in terms])
        all_terms = " + ".join(
            t for t in [main_terms, quad_terms, cubic_terms, interaction_terms] if t
        )
    else:
        all_terms = " + ".join(
            t for t in [main_terms, quad_terms, interaction_terms] if t
        )

    formula = f"Accuracy ~ {all_terms}"
    lm_model = smf.ols(formula, data=df_surrogate).fit()

    def f_to_min(x):
        row = {hp: x[i] for i, hp in enumerate(hyperparameters)}
        df = pd.DataFrame([row])
        return lm_model.predict(df).iloc[0]

    bounds = [(-1.0, 1.0)] * n_hyp

    start_vals = [(0)] * n_hyp

    res = minimize(
        f_to_min,
        x0=start_vals,
        bounds=bounds,
        method="L-BFGS-B"
    )

    doe_pars_coded = res.x
    doe_vals = []
    for i in range(n_hyp):
        v = util_functions.design_log_rescaler(
            [doe_pars_coded[i]],
            lows[i],
            highs[i],
            logList[i]
        )[0]
        if round_param[i]:
            v = int(round(v))
        doe_vals.append(v)

    doe_params = dict(zip(hyperparameters, doe_vals))

    doe_model = base_model(**doe_params)

    doe_model.fit(Xtrain, ytrain)

    yhat_doe = doe_model.predict(Xtest)
    doe_test_acc = balanced_accuracy_score(ytest, yhat_doe)
    doe_val_acc = cross_val_score(
        doe_model,
        Xtrain,
        ytrain,
        scoring="balanced_accuracy",
        cv=cv
    ).mean()

    return doe_test_acc, doe_val_acc
