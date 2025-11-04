#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import env_bootstrap, raw_intake, template_check, session_check, join_check, label_check, vocab_check, window_check, baseline_check, report, docs
from validation.utils import set_seed


def run_task(task_name):
    set_seed(1337)
    
    tasks = {
        'env': env_bootstrap.run,
        'raw': raw_intake.run,
        'templates': template_check.run,
        'sessions': session_check.run,
        'join': join_check.run,
        'labels': label_check.run,
        'vocab': vocab_check.run,
        'windows': window_check.run,
        'baseline': baseline_check.run,
        'report': report.run,
        'docs': docs.run
    }
    
    if task_name == 'all':
        results = {}
        hard_fail = False
        
        for name, func in tasks.items():
            try:
                result = func()
                if isinstance(result, dict):
                    if result.get('hard_fail'):
                        hard_fail = True
                        print(f"Task {name}: HARD FAIL detected", file=sys.stderr)
                        if result.get('issues'):
                            for issue in result.get('issues', []):
                                print(f"  - {issue}", file=sys.stderr)
                    if 'error' in result:
                        print(f"Task {name}: Error - {result['error']}", file=sys.stderr)
                    else:
                        paths = [v for k, v in result.items() if k.endswith('_path')]
                        if paths:
                            print(f"Task {name} ✓ Done: {', '.join(paths)}")
                else:
                    print(f"Task {name} ✓ Done")
            except Exception as e:
                print(f"Task {name}: Exception - {str(e)}", file=sys.stderr)
        
        if hard_fail:
            print("\nHARD FAIL detected in join_check. Please review artifacts/validation/join_integrity.tsv", file=sys.stderr)
            sys.exit(2)
    
    else:
        if task_name not in tasks:
            print(f"Unknown task: {task_name}", file=sys.stderr)
            sys.exit(1)
        
        try:
            result = tasks[task_name]()
            if isinstance(result, dict):
                if result.get('hard_fail'):
                    print(f"Task {task_name}: HARD FAIL detected", file=sys.stderr)
                    if result.get('issues'):
                        for issue in result.get('issues', []):
                            print(f"  - {issue}", file=sys.stderr)
                    print("Please review artifacts/validation/join_integrity.tsv", file=sys.stderr)
                    sys.exit(2)
                if 'error' in result:
                    print(f"Task {task_name}: Error - {result['error']}", file=sys.stderr)
                    sys.exit(1)
                else:
                    paths = [v for k, v in result.items() if k.endswith('_path')]
                    if paths:
                        print(f"Task {task_name} ✓ Done: {', '.join(paths)}")
                    else:
                        print(f"Task {task_name} ✓ Done")
            else:
                print(f"Task {task_name} ✓ Done")
        except Exception as e:
            print(f"Task {task_name}: Exception - {str(e)}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='DeepLog Data Validation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    run_parser = subparsers.add_parser('run', help='Run validation task')
    run_parser.add_argument('--task', required=True, choices=['all', 'env', 'raw', 'templates', 'sessions', 'join', 'labels', 'vocab', 'windows', 'baseline', 'report', 'docs'], help='Task to run')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_task(args.task)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

