# src/features/experiment_logger.py
import csv
import json
from datetime import datetime
import os

# Fieldnames cố định, thứ tự ổn định
FIELDNAMES = [
    "Timestamp", "Model", "Feature_Engineering", "Parameter", "K-Fold",
    "F1", "Accuracy", "Recall", "Prec", "Standard Deviation",
    "Kaggle Score", "Author"
]

def _serialize_params(params):
    # serialize params thành 1 dòng JSON (không xuống dòng)
    try:
        return json.dumps(params, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        return str(params)

def log_experiment(
    output_path,
    model_name,
    feature_name,
    params,
    kfold,
    f1,
    acc,
    rec,
    prec,
    std,
    author="Unknown",
    kaggle_score=None
):
    """
    Safe CSV append with fixed field order and single-line param serialization.
    """
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Feature_Engineering": feature_name,
        "Parameter": _serialize_params(params),
        "K-Fold": kfold,
        "F1": round(float(f1), 4) if f1 is not None else "",
        "Accuracy": round(float(acc), 4) if acc is not None else "",
        "Recall": round(float(rec), 4) if rec is not None else "",
        "Prec": round(float(prec), 4) if prec is not None else "",
        "Standard Deviation": round(float(std), 4) if std is not None else "",
        "Kaggle Score": kaggle_score or "",
        "Author": author,
    }

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    # Mở file với newline='' và encoding
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
        # Nếu file rỗng thì ghi header
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"Logged experiment to {output_path}")
