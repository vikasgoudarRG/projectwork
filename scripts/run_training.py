"""
Main runner script for DeepLog training and evaluation pipeline.

Usage:
    python -m scripts.run_training --task key
    python -m scripts.run_training --task value
    python -m scripts.run_training --task both
    python -m scripts.run_training --task detect --k_sigma 3.0
    python -m scripts.run_training --task eval
    python -m scripts.run_training --task visual
    python -m scripts.run_training --task online
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    parser = argparse.ArgumentParser(description='DeepLog Training & Evaluation Pipeline')
    parser.add_argument('--task', type=str, required=True,
                        choices=['key', 'value', 'both', 'detect', 'eval', 'visual', 'online'],
                        help='Task to run')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (optional)')
    parser.add_argument('--k_sigma', type=float, default=3.0,
                        help='Threshold multiplier for value anomaly detection (default: 3.0)')
    parser.add_argument('--data_path', type=str, default='Event_traces.csv',
                        help='Path to Event_traces.csv')
    parser.add_argument('--h', type=int, default=10,
                        help='Window size (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"DeepLog Pipeline - Task: {args.task.upper()}")
    print("="*70)
    
    if args.task == 'key':
        # Train Key LSTM
        from src.training.train_key import train_key_model
        print("\n[KEY MODEL TRAINING]")
        train_key_model(
            data_path=args.data_path,
            h=args.h,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        print("\n✓ Task Done: Key LSTM training completed")
        print("  Output paths:")
        print("    - models/deeplog_key_model.pt")
        print("    - artifacts/training/key_history.json")
        print("    - artifacts/training/key_loss_curve.png")
    
    elif args.task == 'value':
        # Train Value LSTM
        from src.training.train_value import train_value_model
        print("\n[VALUE MODEL TRAINING]")
        train_value_model(
            data_path=args.data_path,
            h=args.h,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        print("\n✓ Task Done: Value LSTM training completed")
        print("  Output paths:")
        print("    - models/deeplog_value_model.pt")
        print("    - artifacts/training/value_history.json")
        print("    - artifacts/training/value_loss_curve.png")
        print("    - artifacts/training/value_norm.json")
        print("    - artifacts/training/value_threshold.json")
    
    elif args.task == 'both':
        # Train both models
        from src.training.train_key import train_key_model
        from src.training.train_value import train_value_model
        
        print("\n[1/2] KEY MODEL TRAINING")
        train_key_model(
            data_path=args.data_path,
            h=args.h,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        
        print("\n[2/2] VALUE MODEL TRAINING")
        train_value_model(
            data_path=args.data_path,
            h=args.h,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        
        print("\n✓ Task Done: Both models trained")
        print("  Output paths:")
        print("    - models/deeplog_key_model.pt")
        print("    - models/deeplog_value_model.pt")
        print("    - artifacts/training/")
    
    elif args.task == 'detect':
        # Run detection
        from src.detection.detect import detect_anomalies
        print(f"\n[ANOMALY DETECTION] (k_sigma={args.k_sigma})")
        detect_anomalies(
            data_path=args.data_path,
            h=args.h,
            batch_size=args.batch_size,
            g=9,
            k_sigma=args.k_sigma
        )
        print("\n✓ Task Done: Detection completed")
        print("  Output paths:")
        print("    - artifacts/detection/predictions.csv")
        print("    - artifacts/detection/session_anomalies.csv")
        print("    - artifacts/detection/detection_stats.json")
    
    elif args.task == 'eval':
        # Run evaluation
        from src.eval.evaluate import evaluate_detection
        print("\n[EVALUATION]")
        evaluate_detection()
        print("\n✓ Task Done: Evaluation completed")
        print("  Output paths:")
        print("    - artifacts/eval/classification_report.md")
        print("    - artifacts/eval/f1_latency_table.csv")
        print("    - artifacts/eval/metrics.json")
        print("    - graphs/f1_comparison_bar.png")
        print("    - graphs/roc_curve.png")
    
    elif args.task == 'visual':
        # Generate visualization
        from src.visual.workflow_visualizer import visualize_workflow
        print("\n[WORKFLOW VISUALIZATION]")
        visualize_workflow()
        print("\n✓ Task Done: Visualization completed")
        print("  Output paths:")
        print("    - artifacts/visual/workflow_graph.png")
        print("    - artifacts/visual/workflow_stats.json")
    
    elif args.task == 'online':
        # Online learning
        from src.online.update import online_finetune
        print("\n[ONLINE FINE-TUNING]")
        online_finetune(epochs=2, lr=1e-4)
        print("\n✓ Task Done: Online fine-tuning completed")
        print("  Output paths:")
        print("    - models/deeplog_key_model_ft.pt")
        print("    - artifacts/online/update_log.json")
    
    print("\n" + "="*70)
    print("Pipeline execution completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
