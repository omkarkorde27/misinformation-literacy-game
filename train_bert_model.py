"""
Enhanced script to train the improved BERT-based fake news detection model.
This script incorporates the improvements to prevent overfitting and achieve better accuracy.
"""

import argparse
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Import the enhanced model
from bert_fake_news_detection import BERTFakeNewsDetector, EnsembleFakeNewsDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the enhanced BERT fake news detection model')
    parser.add_argument('--dataset', type=str, default='merged_dataset.csv',
                      help='Path to the training dataset (default: merged_dataset.csv)')
    parser.add_argument('--output', type=str, default='enhanced_fake_news_model',
                      help='Directory to save the trained model (default: enhanced_fake_news_model)')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training (default: 16)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of the dataset to use for testing (default: 0.2)')
    parser.add_argument('--max-length', type=int, default=128,
                      help='Maximum token length for BERT (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate (default: 2e-5)')
    parser.add_argument('--cross-validation', action='store_true',
                      help='Use cross-validation for training (default: False)')
    parser.add_argument('--n-folds', type=int, default=5,
                      help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--no-augmentation', action='store_true',
                      help='Disable data augmentation (default: False)')
    parser.add_argument('--patience', type=int, default=3,
                      help='Early stopping patience (default: 3)')
    parser.add_argument('--model-type', type=str, default='bert', choices=['bert', 'roberta'],
                      help='Model type to use (default: bert)')
    parser.add_argument('--no-features', action='store_true',
                      help='Disable additional statistical features (default: False)')
    parser.add_argument('--ensemble', action='store_true',
                      help='Train multiple models for ensemble (default: False)')
    parser.add_argument('--ensemble-size', type=int, default=3,
                      help='Number of models in ensemble (default: 3)')
    
    args = parser.parse_args()
    
    print(f"Training Enhanced {args.model_type.upper()} model using dataset: {args.dataset}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max token length: {args.max_length}")
    print(f"  - Test size: {args.test_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Cross-validation: {args.cross_validation}")
    if args.cross_validation:
        print(f"  - Number of folds: {args.n_folds}")
    print(f"  - Data augmentation: {not args.no_augmentation}")
    print(f"  - Early stopping patience: {args.patience}")
    print(f"  - Using statistical features: {not args.no_features}")
    print(f"  - Using ensemble: {args.ensemble}")
    if args.ensemble:
        print(f"  - Ensemble size: {args.ensemble_size}")
    print(f"Model will be saved to: {args.output}")
    
    if args.ensemble:
        # Train ensemble of models
        train_ensemble(args)
    else:
        # Train single model
        train_single_model(args)

def train_single_model(args):
    """Train a single enhanced BERT model"""
    # Initialize detector with specified parameters
    detector = BERTFakeNewsDetector(
        model_type=args.model_type,
        max_length=args.max_length,
        use_features=not args.no_features
    )
    
    # Train the model
    success = detector.train(
        args.dataset, 
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        use_cross_validation=args.cross_validation,
        n_folds=args.n_folds,
        use_augmentation=not args.no_augmentation,
        patience=args.patience,
        use_features=not args.no_features
    )
    
    if success:
        # Save the trained model
        saved = detector.save_model(args.output)
        if saved:
            print(f"Model successfully trained and saved to {args.output}")
            
            # Test the model on sample news
            test_model_on_samples(detector)
        else:
            print("Failed to save the trained model")
    else:
        print("Failed to train the model")

def train_ensemble(args):
    """Train multiple models for ensemble prediction"""
    models = []
    
    # Create directory for ensemble models
    ensemble_dir = os.path.join(args.output, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # Train models with different configurations
    for i in range(args.ensemble_size):
        print(f"\n===== Training Ensemble Model {i+1}/{args.ensemble_size} =====")
        
        # Vary model parameters for diversity
        if i == 0:
            model_type = "bert"
            learning_rate = args.learning_rate
            max_length = args.max_length
        elif i == 1:
            model_type = "roberta"
            learning_rate = args.learning_rate * 0.8
            max_length = args.max_length
        else:
            model_type = "bert" if i % 2 == 0 else "roberta"
            learning_rate = args.learning_rate * (0.7 + 0.3 * (i / args.ensemble_size))
            max_length = int(args.max_length * (0.8 + 0.4 * (i / args.ensemble_size)))
        
        # Initialize detector
        model_dir = os.path.join(ensemble_dir, f"model_{i+1}")
        detector = BERTFakeNewsDetector(
            model_type=model_type,
            max_length=max_length,
            use_features=not args.no_features
        )
        
        # Train with slightly different parameters
        random_seed = 42 + i * 10
        success = detector.train(
            args.dataset,
            batch_size=args.batch_size,
            epochs=max(3, args.epochs - 1),  # Slightly fewer epochs for ensemble models
            learning_rate=learning_rate,
            test_size=args.test_size,
            use_cross_validation=args.cross_validation,
            n_folds=args.n_folds,
            use_augmentation=not args.no_augmentation,
            random_state=random_seed,
            patience=args.patience,
            use_features=not args.no_features
        )
        
        if success:
            # Save the trained model
            saved = detector.save_model(model_dir)
            if saved:
                print(f"Ensemble model {i+1} successfully trained and saved to {model_dir}")
                models.append(detector)
            else:
                print(f"Failed to save ensemble model {i+1}")
        else:
            print(f"Failed to train ensemble model {i+1}")
    
    # Create ensemble detector if we have multiple models
    if len(models) > 1:
        ensemble = EnsembleFakeNewsDetector(models)
        
        # Test the ensemble on samples
        print("\n===== Testing Ensemble Model =====")
        test_model_on_samples(ensemble)
        
        # Save ensemble configuration
        ensemble_config = {
            "model_count": len(models),
            "model_paths": [os.path.join(ensemble_dir, f"model_{i+1}") for i in range(len(models))],
            "model_types": [model.model_type for model in models]
        }
        
        import json
        with open(os.path.join(ensemble_dir, "ensemble_config.json"), "w") as f:
            json.dump(ensemble_config, f)
        
        print(f"Ensemble configuration saved to {os.path.join(ensemble_dir, 'ensemble_config.json')}")
    else:
        print("Not enough successful models for ensemble prediction")

def test_model_on_samples(detector):
    """Test the trained model on some sample news headlines"""
    print("\nTesting model on sample headlines:")
    samples = [
        # Fake news samples (should be classified as fake)
        "BREAKING: Scientists discover miracle cure for all diseases - Government hiding the truth!",
        "EXCLUSIVE: Shocking evidence reveals politicians are actually lizard people in disguise",
        "You won't believe what this celebrity did to lose 50 pounds in just one week!",
        
        # Real news samples (should be classified as real)
        "Senate passes infrastructure bill with bipartisan support after months of negotiation",
        "Research shows moderate exercise linked to improved cardiovascular health",
        "Technology company announces new smartphone with improved battery life"
    ]
    
    for sample in samples:
        label, confidence = detector.predict(sample)
        print(f"\nHeadline: {sample}")
        print(f"Prediction: {label} (Confidence: {confidence:.2%})")
        
        # For features-enabled models, show feature importance
        if hasattr(detector, 'get_feature_importance') and hasattr(detector, 'use_features') and detector.use_features:
            try:
                importance = detector.get_feature_importance(sample)
                print("Feature importance:")
                for feature, score in importance.items():
                    print(f"  - {feature}: {score:.4f}")
            except:
                pass

if __name__ == "__main__":
    main()