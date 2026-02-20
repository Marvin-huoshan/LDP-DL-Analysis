#!/usr/bin/env python
"""
Attacker Detection CLI - Train and evaluate attacker detection models.

Usage:
    python main.py --data-path /path/to/dataset.csv --model mlp
    python main.py --data-path data.csv --model mlp --epochs 10 --lr 0.0005
"""

import argparse
import os
import torch
import pandas as pd

from config import (
    TRAINING_FEATURES,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT,
    DEFAULT_TEST_SIZE,
    DEFAULT_SEED,
    DATASET_TYPES,
)
from attacker_detector.models import get_model
from attacker_detector.data import load_data, prepare_data, prepare_data_by_dataset_type
from attacker_detector.training import Trainer
from attacker_detector.analysis import run_sensitivity_analysis, plot_sensitivity_metric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate attacker detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to CSV dataset'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['mlp', 'gan', 'attention'],
        help='Model type to use'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DEFAULT_DROPOUT,
        help='Dropout rate'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=DEFAULT_TEST_SIZE,
        help='Test set fraction'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save model and plots'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip sensitivity plots'
    )
    
    # Generalizability training settings
    parser.add_argument(
        '--training-method',
        dest= 'training_method',
        type=str,
        default='none',
        choices=['none', 'cross', 'three-way'],
        help=(
            'Training method: '
            'none = conventional (random split across all dataset types), '
            'cross = train on one dataset_type and test on another, '
            'three-way = train on one, test on another, evaluate on a third'
        )
    )
    parser.add_argument(
        '--train-dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type used for training (required for cross / three-way)'
    )
    parser.add_argument(
        '--test-dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type used for testing (required for cross / three-way)'
    )
    parser.add_argument(
        '--eval-dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type used for evaluation (required for three-way)'
    )
    
    args = parser.parse_args()
    
    # Validate training-method dependent arguments
    if args.training_method in ('cross', 'three-way'):
        if not args.train_dataset or not args.test_dataset:
            parser.error(
                f"--training-method={args.training_method} requires "
                "both --train-dataset and --test-dataset"
            )
        if args.train_dataset == args.test_dataset:
            parser.error(
                "--train-dataset and --test-dataset must be different"
            )
    
    if args.training_method == 'three-way':
        if not args.eval_dataset:
            parser.error(
                "--training-method=three-way requires --eval-dataset"
            )
        if args.eval_dataset in (args.train_dataset, args.test_dataset):
            parser.error(
                "--eval-dataset must differ from --train-dataset and --test-dataset"
            )
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading data from: {args.data_path}")
    df = load_data(args.data_path)
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nTraining features ({len(TRAINING_FEATURES)}): {TRAINING_FEATURES}")
    print(f"Training method: {args.training_method}")
    
    if args.training_method == 'none':
        # Conventional: random stratified split across all dataset types
        print("\nPreparing data (conventional random split)...")
        X_train, X_test, y_train, y_test, scaler, test_indices = prepare_data(
            df,
            TRAINING_FEATURES,
            test_size=args.test_size,
            random_state=args.seed
        )
    else:
        # Cross or three-way: filter by dataset_type
        eval_type = args.eval_dataset if args.training_method == 'three-way' else None
        print(f"\nPreparing data (train={args.train_dataset}, test={args.test_dataset}"
              f"{f', eval={args.eval_dataset}' if eval_type else ''})...")
        
        data = prepare_data_by_dataset_type(
            df,
            TRAINING_FEATURES,
            train_type=args.train_dataset,
            test_type=args.test_dataset,
            eval_type=eval_type,
            random_state=args.seed
        )
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        test_indices = data['test_indices']
        scaler = data['scaler']
    
    print(f"\nCreating {args.model} model...")
    model = get_model(
        args.model,
        input_dim=len(TRAINING_FEATURES),
        dropout_rate=args.dropout
    )
    print(model)
    
    trainer = Trainer(model, device, learning_rate=args.lr, model_type=args.model, epochs=args.epochs)
    trainer.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    # descriptive run name
    global run_name
    run_name = f"{args.model}_ep_{args.epochs}_lr_{args.lr}_bs_{args.batch_size}_drop_{args.dropout}_tm_{args.training_method}"
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f'model_{run_name}.pt')
        trainer.save(model_path)
    
    test_label = f"Test ({args.test_dataset})" if args.training_method != 'none' else "Test"
    trainer.evaluate(X_test, y_test, label=test_label)
    
    print("\nRunning Sensitivity Analysis on test set...")
    df_test = df.iloc[test_indices].reset_index(drop=True)
    
    sensitivity_df = run_sensitivity_analysis(
        model,
        df_test,
        scaler,
        TRAINING_FEATURES,
        device,
        batch_size=4096
    )
    
    print("\nSensitivity Analysis Results (Test):")
    print(sensitivity_df.to_string(index=False))
    
    if args.output_dir:
        results_path = os.path.join(args.output_dir, f'sensitivity_test_results_{run_name}.csv')
        sensitivity_df.to_csv(results_path, index=False)
        print(f"\nTest results saved to: {results_path}")
    
    if not args.no_plot:
        _save_sensitivity_plots(sensitivity_df, 'test', args.output_dir)
    
    if args.training_method == 'three-way':
        X_eval = data['X_eval']
        y_eval = data['y_eval']
        eval_indices = data['eval_indices']
        
        trainer.evaluate(X_eval, y_eval, label=f"Eval ({args.eval_dataset})")
        
        print("\nRunning Sensitivity Analysis on eval set...")
        df_eval = df.iloc[eval_indices].reset_index(drop=True)
        
        eval_sensitivity_df = run_sensitivity_analysis(
            model,
            df_eval,
            scaler,
            TRAINING_FEATURES,
            device,
            batch_size=4096
        )
        
        print("\nSensitivity Analysis Results (Eval):")
        print(eval_sensitivity_df.to_string(index=False))
        
        if args.output_dir:
            eval_results_path = os.path.join(args.output_dir, f'sensitivity_eval_results_{run_name}.csv')
            eval_sensitivity_df.to_csv(eval_results_path, index=False)
            print(f"\nEval results saved to: {eval_results_path}")
        
        if not args.no_plot:
            _save_sensitivity_plots(eval_sensitivity_df, 'eval', args.output_dir)
    
    print("\nDone!")


def _save_sensitivity_plots(sensitivity_df, split_name, output_dir):
    """Save sensitivity plots for a given split (test or eval)."""
    metrics = ['F1_Score', 'Accuracy', 'Precision', 'Recall']
    
    for metric in metrics:
        print(f"\nPlotting {metric.replace('_', ' ')} ({split_name})...")
        save_path = None
        if output_dir:
            save_path = os.path.join(
                output_dir,
                f'sensitivity_{split_name}_{metric.lower()}_{run_name}.png'
            )
        plot_sensitivity_metric(sensitivity_df, metric=metric, save_path=save_path)


if __name__ == '__main__':
    main()
