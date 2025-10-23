from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd

def tune_xgboost(X, y, cv_splits=5, scoring='accuracy', n_jobs=-1, verbose=1):
    """
    Hàm tự động tuning XGBClassifier bằng GridSearchCV.
    ---------------------------------------------------
    Params:
        X, y: Dữ liệu đầu vào
        cv_splits: số lượng fold trong cross-validation
        scoring: metric đánh giá (accuracy, f1, recall, precision, roc_auc,...)
        n_jobs: số CPU dùng để chạy song song (-1 = tất cả)
        verbose: mức độ hiển thị log
        
    Return:
        model_best: mô hình tốt nhất (đã fit)
        results_df: DataFrame chứa toàn bộ kết quả tuning
    """
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

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
