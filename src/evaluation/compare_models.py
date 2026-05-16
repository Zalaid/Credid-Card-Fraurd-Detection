import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR    = os.path.join('models')
CHARTS_DIR    = os.path.join('data', 'processed', 'charts')

MODEL_FILES = {
    "Logistic Regression" : "logistic_regression.pkl",
    "Decision Tree"       : "decision_tree.pkl",
    "Random Forest"       : "random_forest.pkl",
    "Extra Trees"         : "extra_trees.pkl",
    "AdaBoost"            : "adaboost.pkl",
    "Gradient Boosting"   : "gradient_boosting.pkl",
    "XGBoost"             : "xgboost.pkl",
    "LightGBM"            : "lightgbm.pkl",
    "CatBoost"            : "catboost.pkl",
}


def load_test_data():
    X_test = joblib.load(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, 'y_test.pkl'))
    return X_test, y_test


def score_all_models(X_test, y_test):
    results = []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        model = joblib.load(path)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model"     : name,
            "AUC-ROC"   : round(roc_auc_score(y_test, y_proba), 4),
            "F1"        : round(f1_score(y_test, y_pred), 4),
            "Precision" : round(precision_score(y_test, y_pred), 4),
            "Recall"    : round(recall_score(y_test, y_pred), 4),
            "Accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        })
        print(f"  Scored : {name}")
    df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def plot_bar_comparison(df):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    metrics = ["AUC-ROC", "F1", "Precision", "Recall"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))

    for i, metric in enumerate(metrics):
        vals   = df.sort_values(metric, ascending=True)
        bars   = axes[i].barh(vals["Model"], vals[metric], color=colors)
        axes[i].set_title(metric, fontsize=13, fontweight='bold')
        axes[i].set_xlim(max(0, vals[metric].min() - 0.05), 1.0)
        for bar, val in zip(bars, vals[metric]):
            axes[i].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center', fontsize=8)

    plt.suptitle('Model Benchmark — All Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, 'metric_comparison.png')
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  Saved  : {path}")


def plot_roc_curves(X_test, y_test):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_FILES)))

    for (name, fname), color in zip(MODEL_FILES.items(), colors):
        model   = joblib.load(os.path.join(MODELS_DIR, fname))
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})", color=color, linewidth=1.8)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, 'roc_curves.png')
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  Saved  : {path}")


def save_best_model(df):
    best_name = df.iloc[0]["Model"]
    best_file = MODEL_FILES[best_name]
    best_model = joblib.load(os.path.join(MODELS_DIR, best_file))
    out_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, out_path)
    print(f"  Best   : {best_name}  (AUC-ROC={df.iloc[0]['AUC-ROC']})")
    print(f"  Saved  : {out_path}")
    print(f"  Note   : Run tune_catboost.py to get the tuned version → catboost_final.pkl")
    return best_name, best_model


def run():
    print("=" * 55)
    print("  Model Comparison & Selection")
    print("=" * 55)

    X_test, y_test = load_test_data()

    print("\n[1] Scoring all models ...")
    df = score_all_models(X_test, y_test)

    print("\n[2] Generating comparison bar charts ...")
    plot_bar_comparison(df)

    print("\n[3] Generating ROC curves ...")
    plot_roc_curves(X_test, y_test)

    print("\n[4] Saving best model ...")
    best_name, _ = save_best_model(df)

    print("\n")
    print("=" * 55)
    print("  Benchmark Results (ranked by AUC-ROC)")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55)

    return df


if __name__ == '__main__':
    run()
