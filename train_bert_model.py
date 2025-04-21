"""
Script to train the improved BERT-based fake news detection model without overfitting.
This allows you to pre-train the model before deploying the web application.
"""

import argparse
import os
from bert_fake_news_detection import BERTFakeNewsDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the BERT fake news detection model')
    parser.add_argument('--dataset', type=str, default='merged_dataset.csv',
                      help='Path to the training dataset (default: merged_dataset.csv)')
    parser.add_argument('--output', type=str, default='bert_fake_news_model',
                      help='Directory to save the trained model (default: bert_fake_news_model)')
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
    
    args = parser.parse_args()
    
    print(f"Training BERT model using dataset: {args.dataset}")
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
    print(f"Model will be saved to: {args.output}")
    
    # Initialize detector with specified max_length
    detector = BERTFakeNewsDetector(max_length=args.max_length)
    
    # Train the model
    success = detector.train(
        args.dataset, 
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        use_cross_validation=args.cross_validation,
        n_folds=args.n_folds,
        use_augmentation=not args.no_augmentation
    )
    
    if success:
        # Save the trained model
        saved = detector.save_model(args.output)
        if saved:
            print(f"Model successfully trained and saved to {args.output}")
            
            # Test the model on some sample headlines
            print("\nTesting model on sample headlines:")
            sample_headlines = [
                "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",
                "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",
                "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",
                "Trump administration issues new rules on U.S. visa waivers"
            ]
            
            for headline in sample_headlines:
                label, confidence = detector.predict(headline)
                print(f"\nHeadline: {headline}")
                print(f"Prediction: {label} (Confidence: {confidence:.2%})")
        else:
            print("Failed to save the trained model")
    else:
        print("Failed to train the model")

if __name__ == "__main__":
    main()