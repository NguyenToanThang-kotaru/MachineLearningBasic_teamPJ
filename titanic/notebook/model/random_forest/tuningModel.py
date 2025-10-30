from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def tune_random_forest(X, y, cv_splits=5, scoring='accuracy', n_jobs=-1, verbose=1):
    """
    Hàm tự động tuning RandomForestClassifier bằng GridSearchCV.
    -----------------------------------------------------------
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
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
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


