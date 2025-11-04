"""
Evaluate DeepLog performance.
Compute Precision, Recall, F1, Accuracy, and latency metrics.
Generate classification report, F1 comparison, and ROC curve.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc
)


def evaluate_detection():
    """Evaluate detection results."""
    # Create output directories
    os.makedirs('artifacts/eval', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)
    
    print("\n[1/4] Loading detection results...")
    
    # Load session-level predictions
    session_df = pd.read_csv('artifacts/detection/session_anomalies.csv')
    
    # Load detection stats
    with open('artifacts/detection/detection_stats.json', 'r') as f:
        stats = json.load(f)
    
    y_true = session_df['true_label'].values
    y_pred = session_df['fused_anomaly'].values
    
    print(f"  Total sessions: {len(y_true)}")
    print(f"  True anomalies: {sum(y_true)}")
    print(f"  Predicted anomalies: {sum(y_pred)}")
    
    # Compute metrics
    print("\n[2/4] Computing metrics...")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = (y_true == y_pred).mean()
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'latency_ms_per_log': stats['latency_ms_per_log']
    }
    
    print(f"\n  Metrics:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    Latency:   {stats['latency_ms_per_log']:.4f} ms/log")
    
    # Save classification report
    print("\n[3/4] Generating reports...")
    
    report_text = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'])
    
    with open('artifacts/eval/classification_report.md', 'w') as f:
        f.write("# DeepLog Classification Report\n\n")
        f.write("## Summary Metrics\n\n")
        f.write(f"- **Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Precision**: {precision:.4f}\n")
        f.write(f"- **Recall**: {recall:.4f}\n")
        f.write(f"- **F1 Score**: {f1:.4f}\n")
        f.write(f"- **Latency**: {stats['latency_ms_per_log']:.4f} ms/log\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write(f"- True Positives (TP): {tp}\n")
        f.write(f"- True Negatives (TN): {tn}\n")
        f.write(f"- False Positives (FP): {fp}\n")
        f.write(f"- False Negatives (FN): {fn}\n\n")
        f.write("## Detailed Classification Report\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")
    
    print(f"✓ Saved classification report to artifacts/eval/classification_report.md")
    
    # Save F1/Latency table
    results_table = pd.DataFrame([{
        'Model': 'DeepLog',
        'F1_Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Latency_ms': stats['latency_ms_per_log']
    }])
    
    results_table.to_csv('artifacts/eval/f1_latency_table.csv', index=False)
    print(f"✓ Saved results table to artifacts/eval/f1_latency_table.csv")
    
    # Plot F1 comparison bar chart
    print("\n[4/4] Generating visualizations...")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['DeepLog']
    f1_scores = [f1]
    
    bars = ax.bar(models, f1_scores, color='steelblue', alpha=0.8)
    ax.set_ylabel('F1 Score')
    ax.set_title('DeepLog F1 Score Comparison')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('graphs/f1_comparison_bar.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved F1 comparison to graphs/f1_comparison_bar.png")
    
    # Plot ROC curve (if applicable)
    if len(np.unique(y_pred)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DeepLog (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('graphs/roc_curve.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved ROC curve to graphs/roc_curve.png")
        
        metrics['roc_auc'] = float(roc_auc)
    
    # Save metrics JSON
    with open('artifacts/eval/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to artifacts/eval/metrics.json")
    
    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<20}")
    print("-"*60)
    print(f"{'Accuracy':<20} {accuracy:.4f}")
    print(f"{'Precision':<20} {precision:.4f}")
    print(f"{'Recall':<20} {recall:.4f}")
    print(f"{'F1 Score':<20} {f1:.4f}")
    print(f"{'Latency (ms/log)':<20} {stats['latency_ms_per_log']:.4f}")
    if 'roc_auc' in metrics:
        print(f"{'ROC AUC':<20} {metrics['roc_auc']:.4f}")
    print("="*60)
    
    return metrics


if __name__ == '__main__':
    evaluate_detection()
    print("\n✓ Task Done: Evaluation completed")
    print("  - artifacts/eval/classification_report.md")
    print("  - artifacts/eval/f1_latency_table.csv")
    print("  - artifacts/eval/metrics.json")
    print("  - graphs/f1_comparison_bar.png")
    print("  - graphs/roc_curve.png")
