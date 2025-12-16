import os
from os.path import join
import sys
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# === Thêm đường dẫn để import log_experiment ===
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
from log.experiment_logger import log_experiment

def catboost(*, path_to_log_csv=None, author=None, df=None, df_test=None, name="CatBoostClassifier", name_folder='newFE_folder', name_feature='newFe', print_log=True, save_log=False, save_model=False, save_submission=False):
    X = df.drop('Class', axis=1)
    y = df['Class']
<<<<<<< HEAD
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
=======
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
>>>>>>> 503f8857f961fac2ba8fe5f9fae38428207a2d26
    log_path = path_to_log_csv
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    
    fold_index = 1
    model = CatBoostClassifier(random_state=42, verbose=0)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro')
        rec = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        
        if print_log:
            print(f"\n==== Fold {fold_index} results for {name} ====")
            print(f"Fold {fold_index} -> acc:{acc:.4f} | prec:{prec:.4f} | rec:{rec:.4f} | f1:{f1:.4f}")
        fold_index += 1
        
    mean_acc = np.mean(acc_list)
    mean_prec = np.mean(prec_list)
    mean_f1 = np.mean(f1_list)
    mean_rec = np.mean(rec_list)
    if print_log:
        print("\n==== Mean metrics ====")
        print(f"Acc: {mean_acc:.4f}")
        print(f"Prec: {mean_prec:.4f}")
        print(f"F1: {mean_f1:.4f}")
        print(f"Rec: {mean_rec:.4f}")

    # === Ghi log kết quả vào CSV ===
    if save_log:
        log_experiment(
            output_path=log_path,
            model_name=name,
            feature_name=name_folder,
            params= model.get_params(),
            kfold=5,
            acc=mean_acc,
            prec=mean_prec,
            f1=mean_f1,
            rec=mean_rec,
            author=author
        )
        
    final_model = model
    final_model.fit(X, y)

    # === Dump model ra .pkl ===
    if save_log and save_model:
        # === Huấn luyện lại trên toàn bộ dữ liệu train ===
        final_model = model
        final_model.fit(X, y)
        model_dir = join('..', '..', "log", name_folder, "Model Pickles", name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = join(model_dir, f"{name}_{name_feature}.pkl")
        joblib.dump(final_model, model_path)
        print(f"✅ Model saved to {model_path}")
        df_original = pd.read_csv(join('..', '..', 'data', "raw", "test.csv"))
        ids = df_original["Id"]
    # === Tạo file submission ===
    if save_log and save_model and save_submission:
        X_test = df_test.copy()
        if 'Class' in X_test.columns:
            X_test = X_test.drop(columns=['Class'])

        y_test_pred = final_model.predict(X_test)
        y_test_pred = y_test_pred.ravel()

        submission = pd.DataFrame({
            'Id': ids,  # đảm bảo test có cột này
            'Class': y_test_pred
        })

        sub_dir = join('..', '..', 'data', "submissions", name_folder, name)
        os.makedirs(sub_dir, exist_ok=True)
        submission_path = join(sub_dir, f"submission_{name}_{name_feature}.csv")
        submission.to_csv(submission_path, index=False)
        print(f"📤 Submission file saved to {submission_path}")