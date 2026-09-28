import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.chdir(
    r"user/directory/here"
)

from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from Jfunctions import loaders, sim_functions, util_functions

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from openml import datasets
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import LogNorm



load_id_list = [846]

######## SVC Notes #############
# works for 722, 976
# doesn't for 846, 735
# 735 works with smaller gamma range close to 0

######## Boosting notes #########
# works for 735, 722, 976, 846

nsim = 10
gridSize = 25
n_samps_list = [200]
searchMethod = "Grid"

# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: SVC(probability=True, **kw),
#     hyperparameters=["C", "gamma"],
#     hyperparameter_low=[2**-5, 2**-15],
#     hyperparameter_high=[2**15, 2**3],
#     logList=[True, True],
#     round_param=[False, False]
# )
# for 735
modelLoader = loaders.model_loader_maker(
    model_constructor=lambda **kw: SVC(probability=True, **kw),
    hyperparameters=["C", "gamma"],
    hyperparameter_low=[2**-5, 10**-15],
    hyperparameter_high=[1, 1],
    logList=[True, True],
    round_param=[False, False]
)
# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: SVC(kernel="linear", probability=True, **kw),
#     hyperparameters=["C"],
#     hyperparameter_low=[2**-5],
#     hyperparameter_high=[2**15],
#     logList=[True],
#     round_param=[False]
# )
# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: GradientBoostingClassifier(**kw),
#     hyperparameters=["learning_rate"],
#     hyperparameter_low=[2**-10],
#     hyperparameter_high=[1],
#     logList=[True],
#     round_param=[False]
# )
# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: GradientBoostingClassifier(**kw),
#     hyperparameters=["learning_rate", "n_estimators", "max_depth"],
#     hyperparameter_low=[2**-10, 100, 1],
#     hyperparameter_high=[1, 1000, 5],
#     logList=[True, True, False],
#     round_param=[False, True, True]
# )

# modelLoader = loaders.model_loader_maker(
#     model_constructor=lambda **kw: GradientBoostingClassifier(**kw),
#     hyperparameters=["learning_rate", "n_estimators"],
#     hyperparameter_low=[2**-10, 100],
#     hyperparameter_high=[1, 1000],
#     logList=[True, True],
#     round_param=[False, True]
# )

designLoader = loaders.design_loader_maker(
    criterion="space",
    N=gridSize,
    k=len(modelLoader["hyperparameters"]),
    runDepth=500
)


def run_single_sim(j, df, n_samps, modelLoader, designLoader):

    dataLoader = loaders.data_loader_maker(
        data=df,
        feature_cols=range(df.shape[1] - 1),
        target_col=df.shape[1] - 1,
        n_samps=n_samps
    )

    result = sim_functions.plot_sim(
        dataLoader,
        modelLoader,
        designLoader
    )

    return result

full_sim = []

for i, load_id in enumerate(load_id_list):

    dataset = datasets.get_dataset(load_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    df = pd.concat([X, y], axis=1)
    for _, n_samps in enumerate(n_samps_list):

        sim_hold = Parallel(n_jobs=-1)(
            delayed(run_single_sim)(
                j,
                df,
                n_samps,
                modelLoader,
                designLoader
            )
            for j in tqdm(range(nsim))
        )

        sim_hold = pd.DataFrame(sim_hold)

        sim_hold["n_samps"] = n_samps
        sim_hold["model"] = str(modelLoader["model_constructor"]())
        sim_hold["Id"] = load_id
        sim_hold["n_sim"] = nsim

        full_sim.append(sim_hold)



quad_list = full_sim[0]['quad_model']
cubic_list = full_sim[0]['cubic_model']
df_list = full_sim[0]['df_surrogate']

def plot_quad_cubic_heatmaps(
    quad_models,
    cubic_models,
    dfs,
    x_col,
    y_col,
    n_grid=150,
    cmap="viridis"
):
    """
    Creates heatmaps of model predictions with observed points overlaid.

    quad_models : list of statsmodels results (quadratic)
    cubic_models: list of statsmodels results (cubic)
    dfs         : list of DataFrames
    x_col       : list[str] of length 2 (predictors)
    y_col       : str (response column)
    """

    fig, axes = plt.subplots(2, len(dfs), figsize=(4*len(dfs), 8))
    
    if len(dfs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, df in enumerate(dfs):

        # ---- grid for predictions ----
        x1 = df[x_col[0]]
        x2 = df[x_col[1]]

        x1_lin = np.linspace(x1.min(), x1.max(), n_grid)
        x2_lin = np.linspace(x2.min(), x2.max(), n_grid)
        X1g, X2g = np.meshgrid(x1_lin, x2_lin)

        grid_df = pd.DataFrame({
            x_col[0]: X1g.ravel(),
            x_col[1]: X2g.ravel()
        })

        # ---- predictions ----
        Z_quad = quad_models[i].predict(grid_df).values.reshape(n_grid, n_grid)
        Z_cubic = cubic_models[i].predict(grid_df).values.reshape(n_grid, n_grid)

        # =============================
        # Quadratic heatmap
        # =============================
        ax = axes[0, i]
        im = ax.imshow(
            Z_quad,
            extent=[x1_lin.min(), x1_lin.max(), x2_lin.min(), x2_lin.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap
        )

        ax.scatter(
            x1, x2,
            c=df[y_col],
            cmap=cmap,
            edgecolor="k",
            s=40
        )

        ax.set_title(f"Sim {i+1} — Quadratic")
        ax.set_xlabel(x_col[0])
        ax.set_ylabel(x_col[1])

        # =============================
        # Cubic heatmap
        # =============================
        ax = axes[1, i]
        im = ax.imshow(
            Z_cubic,
            extent=[x1_lin.min(), x1_lin.max(), x2_lin.min(), x2_lin.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap
        )

        ax.scatter(
            x1, x2,
            c=df[y_col],
            cmap=cmap,
            edgecolor="k",
            s=40
        )

        ax.set_title(f"Sim {i+1} — Cubic")
        ax.set_xlabel(x_col[0])
        ax.set_ylabel(x_col[1])

    # ---- shared colorbar ----
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=y_col)

    plt.tight_layout()
    plt.show()

plot_quad_cubic_heatmaps(
    quad_models=quad_list,
    cubic_models=cubic_list,
    dfs=df_list,
    # x_col=["learning_rate", "n_estimators"],
    x_col=["C", "gamma"],
    y_col="Accuracy"
)


# aic_list = []
# for i in range(n_samps):
#     aic_list.append(cubic_list[i].aic)

# np.mean(aic_list)

## 77.15 for gbm
## -220.5 for svm
## not super comparable - gbm experiences way more deviation across the space (more tunable)