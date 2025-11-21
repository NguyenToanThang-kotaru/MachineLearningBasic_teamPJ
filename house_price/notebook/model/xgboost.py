import os
from os.path import join
import sys
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# === Thêm đường dẫn để import log_experiment ===
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
from log.experiment_logger import log_experiment

def xgboost(*, path_to_log_csv = None, author=None, df=None, df_test=None, name="XGBRegressor", name_folder='newFE', save_log=False, save_model=False, save_submission=False):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    log_path = path_to_log_csv
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    r2s = []
    
    fold_index = 1
    model = XGBRegressor()
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = model.score(X_val, y_val)
        
        rmses.append(rmse)
        r2s.append(r2)
        
        print(f"\n==== Fold {fold_index} results for {name} ====")
        print(f"Fold {fold_index} - R2: {r2:.4f} | RMSE: {rmse:.4f}")
        fold_index += 1

    mean_r2 = np.mean(r2s)
    mean_rmse = np.mean(rmses)
    print("\n==== Mean metrics ====")
    print(f"R2 Score: {mean_r2:.4f}")
    print(f"RMSE: {mean_rmse:.4f}")

    # === Ghi log kết quả vào CSV ===
    if save_log:
        log_experiment(
            output_path=log_path,
            model_name=name,
            feature_name=name_folder,
            params= model.get_params(),
            kfold=5,
            rmse=mean_rmse,
            r2=mean_r2,
            author=author
        )

    if save_log and save_model:
        # === Huấn luyện lại trên toàn bộ dữ liệu train ===
        final_model = model
        final_model.fit(X, y)

        # === Dump model ra .pkl ===
        model_dir = join('..', '..', "log", name_folder, "Model Pickles", name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = join(model_dir, f"{name}_{name_folder}.pkl")
        joblib.dump(final_model, model_path)
        print(f"✅ Model saved to {model_path}")
        df_original = pd.read_csv(join('..', '..', 'data', "raw", "test.csv"))
        ids = df_original["Id"]
    if save_log and save_model and save_submission:
        # === Tạo file submission ===
        X_test = df_test.copy()
        if 'SalePrice' in X_test.columns:
            X_test = X_test.drop(columns=['SalePrice'])

        y_test_pred = final_model.predict(X_test)

        submission = pd.DataFrame({
            'Id': ids,  # đảm bảo test có cột này
            'SalePrice': y_test_pred
        })

        sub_dir = join('..', '..', 'data', "submissions", name_folder, name)
        os.makedirs(sub_dir, exist_ok=True)
        submission_path = join(sub_dir, f"submission_{name}_{name_folder}.csv")
        submission.to_csv(submission_path, index=False)
        print(f"📤 Submission file saved to {submission_path}")