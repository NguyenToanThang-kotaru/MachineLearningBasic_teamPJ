from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

def tune_logistic_regression(X, y, cv_splits=5, scoring='accuracy', n_jobs=-1, verbose=1):
    """
    Hàm tự động tuning LogisticRegression bằng GridSearchCV.
    --------------------------------------------------------
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
    
    lr = LogisticRegression(random_state=42, max_iter=1000)

    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],  # saga hỗ trợ l1, l2, elasticnet
        'l1_ratio': [0, 0.5, 1]  # chỉ dùng khi penalty='elasticnet'
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=lr,
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
