from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd

def tune_catboost(X, y, cv_splits=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1):
    """
    Hàm tự động tuning CatBoostRegressor bằng GridSearchCV.
    """

    cb = CatBoostRegressor(
        random_state=42,
        silent=True,
        loss_function='RMSE'   # phù hợp Regression
    )

    param_grid = {
        'iterations': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7]
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=cb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    grid_search.fit(X, y)

    print(f"\n Best Score ({scoring}): {grid_search.best_score_:.4f}")
    print(f" Best Params: {grid_search.best_params_}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    model_best = grid_search.best_estimator_

    return model_best, results_df


def tune_xgboost(X, y, cv_splits=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1):
    """
    Hàm tự động tuning XGBoostRegressor bằng GridSearchCV (cho bài toán dự đoán giá nhà).
    """

    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='auto'
    )

    # Grid search parameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_lambda': [1, 3, 5],
        'reg_alpha': [0, 1, 3]
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    grid_search.fit(X, y)

    print(f"\n Best Score ({scoring}): {grid_search.best_score_:.4f}")
    print(f" Best Params: {grid_search.best_params_}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    model_best = grid_search.best_estimator_

    return model_best, results_df
