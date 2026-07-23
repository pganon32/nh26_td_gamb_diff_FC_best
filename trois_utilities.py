"""Utilities for use with the 2023 NeuroHackademy data showcase.
"""

from io import StringIO
import numpy as np
import pandas as pd

def ls(path):
    "Lists the contents of the given path."
    # If path is not a directory, raise an error:
    if not path.is_dir():
        raise ValueError(f"Path '{path}' is not a directory")
    else:
        return list(path.iterdir())


def crawl(path, indent=0):
    "Prints a nested tree of the contents of the given path."
    print((' '*indent) + path.name)
    if path.is_dir():
        for subpath in path.iterdir():
            crawl(subpath, indent=(indent + 3))
    else:
        pass


def load_aws_credentials(profile_name):
    "Returns (access_key, secred_key) from ~/.aws/credentials for the given profile."
    import boto3
    ses = boto3.Session(profile_name=profile_name)
    creds = ses.get_credentials()
    return (creds.access_key, creds.secret_key)




import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, balanced_accuracy_score, f1_score
)

from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def run_nested_cv_regression(
    df,
    pca_df,
    outcome_col="DDisc_AUC_40K",
    age_col="age_group",
    sex_col="Gender",
    model_name="ridge",
    task="regression",
    use_age_average=False,
    use_age_covariate=True,
    use_sex_covariate=True,
    include_sex_for_age=False,
    n_splits=5,
    random_state=42,
    n_bootstrap=2000,
    ci_level=0.95,
):
    """
    Nested CV for either:
    - regression of outcome_col from PCA (+ age/sex covariates), or
    - classification of age_group from PCA (diffusion parameters).

    task:
    - "regression"
    - "age_classification"
    """

    task = task.lower()
    model_name = model_name.lower()

    if task not in ["regression", "age_classification"]:
        raise ValueError("task must be 'regression' or 'age_classification'")

    if use_age_average and task != "regression":
        raise ValueError("use_age_average=True is only supported for regression")

    # -------------------------
    # Build modeling table
    # -------------------------
    if task == "regression":
        cols = [outcome_col]
    
        if use_age_covariate:
            if use_age_average:
                cols.append("age_average")
            else:
                cols.append(age_col)
    
        if use_sex_covariate:
            cols.append(sex_col)
    
        model_df = pd.concat(
            [
                df[cols],
                pca_df,
            ],
            axis=1,
        )
    
        model_df[outcome_col] = pd.to_numeric(model_df[outcome_col], errors="coerce")
    
        if use_age_covariate and use_age_average:
            model_df["age_average"] = pd.to_numeric(model_df["age_average"], errors="coerce")
    
        if use_age_covariate and not use_age_average:
            model_df = pd.get_dummies(model_df, columns=[age_col], drop_first=True)
    
        if use_sex_covariate:
            model_df = pd.get_dummies(model_df, columns=[sex_col], drop_first=True)
    
        model_df = model_df.dropna()
    
        y = model_df[outcome_col].astype(float)
        X = model_df.drop(columns=[outcome_col]).astype(float)
    
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = "neg_mean_squared_error"

    else:  # age classification
        base_cols = [age_col]
        if include_sex_for_age:
            base_cols.append(sex_col)

        model_df = pd.concat(
            [
                df[base_cols],
                pca_df,
            ],
            axis=1,
        ).dropna()

        if include_sex_for_age:
            model_df = pd.get_dummies(model_df, columns=[sex_col], drop_first=True)

        y = model_df[age_col].astype(str)
        X = model_df.drop(columns=[age_col]).astype(float)

        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = "balanced_accuracy"

    feature_names = X.columns.tolist()

    # -------------------------
    # Model + grid by task
    # -------------------------
    if task == "regression":
        if model_name == "ridge":
            estimator = Ridge()
            param_grid = {"model__alpha": np.logspace(-3, 3, 25)}

        elif model_name in ["elastic", "elasticnet"]:
            estimator = ElasticNet(max_iter=10000)
            param_grid = {
                "model__alpha": np.logspace(-3, 2, 20),
                "model__l1_ratio": np.linspace(0.1, 0.9, 9),
            }

        elif model_name == "lasso":
            estimator = Lasso(max_iter=10000)
            param_grid = {"model__alpha": np.logspace(-3, 2, 20)}

        elif model_name in ["linear", "ols"]:
            estimator = LinearRegression()
            param_grid = {}

        elif model_name == "svr":
            estimator = SVR()
            param_grid = {
                "model__C": np.logspace(-2, 3, 10),
                "model__epsilon": [0.01, 0.05, 0.1, 0.2],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", "auto"],
            }

        elif model_name == "rf":
            estimator = RandomForestRegressor(random_state=random_state)
            param_grid = {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            }

        elif model_name == "gbr":
            estimator = GradientBoostingRegressor(random_state=random_state)
            param_grid = {
                "model__n_estimators": [100, 200, 500],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_split": [2, 5, 10],
            }

        elif model_name == "knn":
            estimator = KNeighborsRegressor()
            param_grid = {
                "model__n_neighbors": [3, 5, 7, 9, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            }

        else:
            raise ValueError(
                "For regression, model_name must be one of: ridge, elastic, elasticnet, lasso, linear, ols, svr, rf, gbr, knn"
            )

    else:  # age classification
        if model_name in ["logreg", "logistic", "logisticregression"]:
            estimator = LogisticRegression(
                max_iter=5000, class_weight="balanced", multi_class="auto"
            )
            param_grid = {"model__C": np.logspace(-3, 3, 20)}

        elif model_name == "rf":
            estimator = RandomForestClassifier(random_state=random_state, class_weight="balanced")
            param_grid = {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            }

        elif model_name in ["svc", "svm"]:
            estimator = SVC(class_weight="balanced")
            param_grid = {
                "model__C": np.logspace(-2, 3, 10),
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", "auto"],
            }

        elif model_name == "knn":
            estimator = KNeighborsClassifier()
            param_grid = {
                "model__n_neighbors": [3, 5, 7, 9, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            }

        else:
            raise ValueError(
                "For age_classification, model_name must be one of: logreg, rf, svc, knn"
            )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=-1,
    )

    # -------------------------
    # Nested CV loop
    # -------------------------
    y_pred_all = np.empty(len(y), dtype=object if task == "age_classification" else float)
    if task == "regression":
        y_pred_all[:] = np.nan
    else:
        y_pred_all[:] = ""

    fold_metrics = []
    feature_weight_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        y_pred_all[test_idx] = y_pred

        if task == "regression":
            fold_metrics.append({
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            })
        else:
            fold_metrics.append({
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
            })

        best_model = search.best_estimator_.named_steps["model"]
        fold_weights = None

        if hasattr(best_model, "coef_"):
            coef = np.asarray(best_model.coef_)
            if coef.ndim == 1:
                fold_weights = coef
            else:
                fold_weights = np.mean(np.abs(coef), axis=0)

        elif hasattr(best_model, "feature_importances_"):
            fold_weights = np.asarray(best_model.feature_importances_)

        if fold_weights is not None:
            feature_weight_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_idx,
                        "feature": feature_names,
                        "weight": np.ravel(fold_weights),
                    }
                )
            )

    # -------------------------
    # Summaries
    # -------------------------
    feature_weights = None
    feature_weights_summary = None

    if feature_weight_rows:
        feature_weights = pd.concat(feature_weight_rows, ignore_index=True)

        feature_weights_summary = (
            feature_weights.groupby("feature")["weight"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .sort_values("mean", key=lambda s: s.abs(), ascending=False)
        )

        rng = np.random.default_rng(random_state)
        alpha = 1.0 - ci_level
        lower_q = 100 * (alpha / 2.0)
        upper_q = 100 * (1.0 - alpha / 2.0)

        bootstrap_rows = []
        for feature in feature_names:
            values = feature_weights.loc[feature_weights["feature"] == feature, "weight"].to_numpy()
            if len(values) == 0:
                continue

            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))

            bootstrap_means = np.asarray(bootstrap_means)
            bootstrap_rows.append(
                {
                    "feature": feature,
                    "bootstrap_mean": np.mean(bootstrap_means),
                    "bootstrap_ci_low": np.percentile(bootstrap_means, lower_q),
                    "bootstrap_ci_high": np.percentile(bootstrap_means, upper_q),
                }
            )

        feature_weights_summary = feature_weights_summary.merge(
            pd.DataFrame(bootstrap_rows), on="feature", how="left"
        )

    if task == "regression":
        fold_r2 = [m["r2"] for m in fold_metrics]
        fold_rmse = [m["rmse"] for m in fold_metrics]

        return {
            "task": task,
            "mean_r2": np.mean(fold_r2),
            "sd_r2": np.std(fold_r2, ddof=1),
            "mean_rmse": np.mean(fold_rmse),
            "sd_rmse": np.std(fold_rmse, ddof=1),
            "fold_r2": fold_r2,
            "fold_rmse": fold_rmse,
            "y_pred": y_pred_all,
            "feature_weights": feature_weights,
            "feature_weights_summary": feature_weights_summary,
        }

    else:
        fold_acc = [m["accuracy"] for m in fold_metrics]
        fold_bacc = [m["balanced_accuracy"] for m in fold_metrics]
        fold_f1 = [m["f1_macro"] for m in fold_metrics]

        return {
            "task": task,
            "mean_accuracy": np.mean(fold_acc),
            "sd_accuracy": np.std(fold_acc, ddof=1),
            "mean_balanced_accuracy": np.mean(fold_bacc),
            "sd_balanced_accuracy": np.std(fold_bacc, ddof=1),
            "mean_f1_macro": np.mean(fold_f1),
            "sd_f1_macro": np.std(fold_f1, ddof=1),
            "fold_accuracy": fold_acc,
            "fold_balanced_accuracy": fold_bacc,
            "fold_f1_macro": fold_f1,
            "y_pred": y_pred_all,
            "feature_weights": feature_weights,
            "feature_weights_summary": feature_weights_summary,
        }

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_diffusion_pca(df, n_pcs=25, pc_substrings=None):
    if pc_substrings is None:
        pc_substrings = ["dki_fa", "dki_md", "dki_mk", "dki_awf"]

    diffusion_cols = [
        col for col in df.columns
        if any(sub in col for sub in pc_substrings)
    ]

    X_diff = df[diffusion_cols].copy()
    X_diff = X_diff.apply(pd.to_numeric, errors="coerce")
    X_diff = X_diff.fillna(X_diff.mean())

    X_scaled = StandardScaler().fit_transform(X_diff)

    pca = PCA(n_components=n_pcs, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
        index=df.index
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=diffusion_cols,
        columns=[f"PC{i+1}" for i in range(n_pcs)]
    )

    explained_variance = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_pcs)],
        "ExplainedVariance": pca.explained_variance_ratio_,
        "CumulativeVariance": pca.explained_variance_ratio_.cumsum()
    })

    return {
        "diffusion_cols": diffusion_cols,
        "pca": pca,
        "pca_df": pca_df,
        "loadings": loadings,
        "explained_variance": explained_variance
    }
