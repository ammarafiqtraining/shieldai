import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import config


def plot_confusion_matrix(cm, model_name, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(fpr, tpr, roc_auc, model_name, save_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.upper()}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(save_dir, f'roc_curve_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def evaluate_model(model, model_name, X_test, y_test, threshold=0.90):
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*50}")
    
    y_pred = model.predict(X_test)
    y_proba = getattr(model, 'predict_proba', lambda x: None)(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    print(f"\nMetrics:")
    for metric, value in metrics.items():
        status = "[PASS]" if metric != "recall" or value >= threshold else "[FAIL]"
        print(f"  {metric.capitalize():12s}: {value:.4f} {status}")
    
    recall_met = metrics["recall"] >= threshold
    print(f"\n  Recall >= {threshold}: {'PASS' if recall_met else 'FAIL'}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        metrics["auc"] = roc_auc
        
        plot_roc_curve(fpr, tpr, roc_auc, model_name, config.MODELS_DIR)
    
    plot_confusion_matrix(cm, model_name, config.MODELS_DIR)
    
    return metrics


def compare_models(results):
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    metrics_to_compare = ["accuracy", "precision", "recall", "f1", "auc"]
    
    header = f"{'Metric':<15}"
    for model_name in results.keys():
        header += f" {model_name.upper():>12}"
    print(header)
    print("-" * (15 + 13 * len(results)))
    
    for metric in metrics_to_compare:
        row = f"{metric.capitalize():<15}"
        for model_name, metrics in results.items():
            value = metrics.get(metric, "N/A")
            if isinstance(value, float):
                row += f" {value:>12.4f}"
            else:
                row += f" {str(value):>12}"
        print(row)
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    
    best_recall_model = max(results.keys(), key=lambda k: results[k].get("recall", 0))
    best_f1_model = max(results.keys(), key=lambda k: results[k].get("f1", 0))
    
    print(f"  Best Recall: {best_recall_model.upper()} ({results[best_recall_model]['recall']:.4f})")
    print(f"  Best F1 Score: {best_f1_model.upper()} ({results[best_f1_model]['f1']:.4f})")
    
    if results[best_recall_model]["recall"] >= config.TARGET_THRESHOLD:
        print(f"\n  [OK] {best_recall_model.upper()} meets >={config.TARGET_THRESHOLD*100:.0f}% recall target")
        print(f"  -> Recommended for production deployment")
    else:
        print(f"\n  [FAIL] No model meets >={config.TARGET_THRESHOLD*100:.0f}% recall target")
        print(f"  -> Consider hyperparameter tuning or additional features")
    
    return results


def main(X_test, y_test, models_dict):
    print("=" * 60)
    print("PHISHING DETECTION - MODEL EVALUATION")
    print("=" * 60)
    
    results = {}
    for model_name, model in models_dict.items():
        results[model_name] = evaluate_model(
            model, model_name, X_test, y_test, 
            threshold=config.TARGET_THRESHOLD
        )
    
    compare_models(results)
    
    eval_info = {
        "version": config.PROCESSED_VERSION,
        "evaluated_at": config.get_timestamp(),
        "test_samples": int(len(y_test)),
        "results": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                        for kk, vv in v.items()} for k, v in results.items()},
        "target_threshold": config.TARGET_THRESHOLD
    }
    
    eval_path = os.path.join(config.MODELS_DIR, "evaluation_info.json")
    with open(eval_path, "w") as f:
        json.dump(eval_info, f, indent=2)
    print(f"\n  Saved: {eval_path}")
    
    print("\n[EVALUATION COMPLETE]")
    return results


if __name__ == "__main__":
    import joblib
    from preprocess import main as preprocess_main
    
    X_train, X_test, y_train, y_test, feature_names, metadata = preprocess_main()
    
    models = {}
    for model_type in ["rf", "lr"]:
        model_path = os.path.join(
            config.MODELS_DIR,
            config.get_model_filename(model_type, config.PROCESSED_VERSION)
        )
        if os.path.exists(model_path):
            models[model_type] = joblib.load(model_path)
    
    if models:
        main(X_test, y_test, models)
    else:
        print("No trained models found. Run train.py first.")
