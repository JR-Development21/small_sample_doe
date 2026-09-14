import numpy as np
import pandas as pd
import itertools

from Jfunctions import util_functions

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, make_scorer
from scipy.optimize import minimize
import statsmodels.formula.api as smf



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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # param grid is for CV
    # param grid cart is for RSM
    # param space is for RSM
    # param grid space is for CV


    ########################### get four grids ###########################

    grid_out = run_grid(Xtrain, ytrain, hyperparameters, round_param, base_model, param_grid_cart_dict, cv)
    space_out = run_grid(Xtrain, ytrain, hyperparameters, round_param, base_model, params_space, cv)

    space_ba_emp_test, space_ba_emp_val = empirical_min(Xtrain, ytrain, Xtest, ytest, base_model,hyperparameters, round_param, space_out, True, cv) 
    space_mse_emp_test, space_mse_emp_val = empirical_min(Xtrain, ytrain, Xtest, ytest, base_model,hyperparameters, round_param, space_out, False, cv) 
    space_mse_rsm_2_test, space_mse_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,space_out, hyperparameters, n_hyp, round_param, logList, lows, highs, False,cv, False) 
    space_ba_rsm_2_test, space_ba_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,space_out, hyperparameters, n_hyp, round_param, logList, lows, highs, False,cv, True) 
    grid_mse_rsm_2_test, grid_mse_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,grid_out, hyperparameters, n_hyp, round_param, logList, lows, highs, False,cv, False) 
    grid_ba_rsm_2_test, grid_ba_rsm_2_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,grid_out, hyperparameters, n_hyp, round_param, logList, lows, highs, False,cv, True) 
    grid_mse_rsm_3_test, grid_mse_rsm_3_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model,grid_out, hyperparameters, n_hyp, round_param, logList, lows, highs, True,cv, False) 
    grid_ba_rsm_3_test, grid_ba_rsm_3_val = expected_min(Xtrain, ytrain, Xtest, ytest, base_model, grid_out, hyperparameters, n_hyp, round_param, logList, lows, highs, True, cv, True)

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
          hyperparameters,
          round_param,
          params_df,
          ba_bool,
          cv
):
    metric_string = "Accuracy" if ba_bool else "mse"
    best_idx = params_df[metric_string].idxmin()
    params = {}
    for i, hp in enumerate(hyperparameters):
        params[hp] = params_df.loc[best_idx][f"model__{hp}"]
        if round_param[i]:
            params[hp] = np.round(params[hp]).astype(int)
    
    best_model = base_model(**params)
    best_model.fit(Xtrain, ytrain)

    yhat_test = best_model.predict(Xtest)

    grid_search_test_acc = balanced_accuracy_score(ytest, yhat_test)
    grid_search_val_acc = cross_val_score(
        best_model,
        Xtrain,
        ytrain,
        scoring="balanced_accuracy",
        cv=cv
    ).mean()
    return grid_search_test_acc, grid_search_val_acc

def run_grid(
          Xtrain,
          ytrain,
          hyperparameters,
          round_param,
          base_model,
          param_grid,
          cv):
    des_grid_real = pd.DataFrame(param_grid)
    des_acc = []
    des_mse = []

    for _, row in des_grid_real.iterrows():
        params = {}
        for i, hp in enumerate(hyperparameters):
            params[hp] = row[f"model__{hp}"]
            if round_param[i]:
                params[hp] = np.round(params[hp]).astype(int)


        model = base_model(**params)

        scores = cross_validate(
            model,
            Xtrain,
            ytrain,
            scoring={
                "balanced_accuracy": "balanced_accuracy",
                "brier_score": neg_brier_scorer
            },
            cv=cv
        )
        des_acc.append(-scores["test_balanced_accuracy"].mean())
        des_mse.append(-scores["test_brier_score"].mean())

    des_grid_real["Accuracy"] = des_acc
    des_grid_real["mse"] = des_mse
    return des_grid_real
    

def expected_min(
          Xtrain,
          ytrain,
          Xtest,
          ytest,
          base_model,
          des_grid_real,
          hyperparameters,
          n_hyp,
          round_param,
          logList,
          lows,
          highs,
          cubic_bool,
          cv,
          ba_bool
):
    metric_string = "Accuracy" if ba_bool else "mse"
    des_scores = des_grid_real[metric_string].values
    df_surrogate = pd.DataFrame()

    for i, hp in enumerate(hyperparameters):
        df_surrogate[hp] = util_functions.design_log_scaler(
            des_grid_real[f"model__{hp}"].values,
            lows[i],
            highs[i],
            logList[i]
        )
    

    df_surrogate[metric_string] = des_scores

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

    formula = f"{metric_string} ~ {all_terms}"
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
