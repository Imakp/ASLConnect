"""
Main application script for ASL recognition project.
Provides a command-line interface to access all functionality.
"""

import os
import sys
import argparse

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from asl_modules.data_collection import collect_asl_data, collect_multiple_asl_letters
from asl_modules.preprocessing import load_and_preprocess_data, analyze_dataset, combine_datasets
from asl_modules.training import train_and_evaluate_model, train_unified_model
from asl_modules.inference import run_asl_recognition, predict_from_saved_landmarks
from asl_modules.utils import get_project_root

def main():
    """
    Main function to provide a command-line interface for the ASL recognition project.
    """
    parser = argparse.ArgumentParser(description='ASL Recognition System')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Collect data command
    collect_parser = subparsers.add_parser('collect', help='Collect ASL hand landmark data')
    collect_parser.add_argument('--letter', type=str, help='ASL letter to collect data for')
    collect_parser.add_argument('--samples', type=int, help='Number of samples to collect')
    collect_parser.add_argument('--output', type=str, default='asl_landmarks_dataset.csv', 
                               help='Output CSV file name')
    collect_parser.add_argument('--multi-hand', action='store_true',
                               help='Collect data for multi-hand signs')
    
    # Analyze data command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze the collected dataset')
    analyze_parser.add_argument('--input', type=str, default='asl_landmarks_dataset.csv', 
                               help='Input CSV file name')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the ASL recognition model')
    train_parser.add_argument('--input', type=str, default='asl_landmarks_dataset.csv', 
                             help='Input CSV file name')
    train_parser.add_argument('--hidden-size', type=int, default=128, 
                             help='Number of neurons in the hidden layer')
    train_parser.add_argument('--max-iter', type=int, default=300, 
                             help='Maximum number of iterations')
    train_parser.add_argument('--multi-hand', action='store_true',
                             help='Train a model for multi-hand signs')
    
    # Run recognition command
    run_parser = subparsers.add_parser('run', help='Run real-time ASL recognition')
    run_parser.add_argument('--model', type=str, default='asl_mlp_multi_hand_model.joblib', 
                           help='Model file name (stored in asl_model directory)')
    run_parser.add_argument('--scaler', type=str, default='hand_landmarks_scaler.joblib', 
                           help='Scaler file name (stored in asl_model directory)')
    run_parser.add_argument('--multi-hand', action='store_true', default=True,
                           help='Use multi-hand model for recognition')
    
    # Evaluate on saved data command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on saved data')
    eval_parser.add_argument('--input', type=str, default='asl_landmarks_dataset.csv', 
                            help='Input CSV file name')
    eval_parser.add_argument('--multi-hand', action='store_true',
                            help='Evaluate using multi-hand model')
    
    # Combine datasets command
    combine_parser = subparsers.add_parser('combine', help='Combine single-hand and multi-hand datasets')
    combine_parser.add_argument('--single-hand', type=str, required=True,
                              help='Single-hand dataset CSV file')
    combine_parser.add_argument('--multi-hand', type=str, required=True,
                              help='Multi-hand dataset CSV file')
    combine_parser.add_argument('--output', type=str, default='combined_asl_dataset.csv',
                              help='Output combined dataset CSV file')
    
    # Train unified model command
    unified_parser = subparsers.add_parser('train-unified', help='Train a unified model for both single and multi-hand signs')
    unified_parser.add_argument('--single-hand', type=str, required=True,
                              help='Single-hand dataset CSV file')
    unified_parser.add_argument('--multi-hand', type=str, required=True,
                              help='Multi-hand dataset CSV file')
    unified_parser.add_argument('--output', type=str, default='unified_asl_dataset.csv',
                              help='Output combined dataset CSV file')
    unified_parser.add_argument('--hidden-size', type=int, default=128,
                              help='Number of neurons in the hidden layer')
    unified_parser.add_argument('--max-iter', type=int, default=300,
                              help='Maximum number of iterations')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'collect':
        if args.letter and args.samples:
            collect_asl_data(args.letter, args.samples, args.output, args.multi_hand)
        else:
            collect_multiple_asl_letters(output_file=args.output)
    
    elif args.command == 'analyze':
        analyze_dataset(args.input)
    
    elif args.command == 'train':
        if args.multi_hand:
            print("Training model for multi-hand signs...")
            # Use a different model filename for multi-hand model
            model_filename = 'asl_mlp_multi_hand_model.joblib'
            train_and_evaluate_model(args.input, args.hidden_size, args.max_iter)
        else:
            print("Training model for single-hand signs...")
            train_and_evaluate_model(args.input, args.hidden_size, args.max_iter)
    
    elif args.command == 'run':
        print("Running recognition...")
        model_filename = args.model
        scaler_filename = args.scaler
        run_asl_recognition(model_filename, scaler_filename)
    
    elif args.command == 'evaluate':
        if args.multi_hand:
            print("Evaluating multi-hand model...")
            model_filename = 'asl_mlp_multi_hand_model.joblib'
            predict_from_saved_landmarks(args.input)
        else:
            predict_from_saved_landmarks(args.input)
    
    elif args.command == 'combine':
        print(f"Combining datasets: {args.single_hand} and {args.multi_hand}...")
        combine_datasets(args.single_hand, args.multi_hand, args.output)
    
    elif args.command == 'train-unified':
        print("Training unified model for both single and multi-hand signs...")
        train_unified_model(
            args.single_hand, 
            args.multi_hand, 
            args.output, 
            args.hidden_size, 
            args.max_iter
        )
    
    else:
        # If no command is provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()