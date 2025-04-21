"""
Script to train the BERT-based fake news detection model without running the web interface.
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
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to use for testing (default: 0.2)')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum token length for BERT (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    
    args = parser.parse_args()
    
    print(f"Training BERT model using dataset: {args.dataset}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max token length: {args.max_length}")
    print(f"  - Test size: {args.test_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"Model will be saved to: {args.output}")
    
    # Initialize and train the model
    detector = BERTFakeNewsDetector()
    success = detector.train(
        args.dataset, 
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        max_length=args.max_length
    )
    
    if success:
        # Save the trained model
        saved = detector.save_model(args.output)
        if saved:
            print(f"Model successfully trained and saved to {args.output}")
        else:
            print("Failed to save the trained model")
    else:
        print("Failed to train the model")

if __name__ == "__main__":
    main()