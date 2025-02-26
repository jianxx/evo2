#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script showcasing how to use the Evo2 fine-tuning framework for sequence classification.
This script demonstrates:
1. Generating synthetic data
2. Training a classifier model
3. Evaluating and visualizing results
4. Making predictions on new data
"""

import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging
import tempfile
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from evo2 import Evo2
    from classifier import (
        Evo2Classifier, 
        Evo2FineTuner, 
        SequenceClassificationDataset,
        collate_fn
    )
except ImportError:
    logger.error("Could not import required modules. Make sure the 'evo2' package is installed.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo of Evo2 fine-tuning for sequence classification")
    
    # Model selection
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="evo2_1b_base", 
        choices=["evo2_1b_base", "evo2_7b", "evo2_7b_base", "evo2_40b", "evo2_40b_base"],
        help="Evo2 model to use (smaller models recommended for the demo)"
    )
    
    # Demo parameters
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./demo_output", 
        help="Directory to save demo outputs"
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=2, 
        help="Number of classes in the demo dataset"
    )
    parser.add_argument(
        "--num_sequences", 
        type=int, 
        default=200, 
        help="Number of sequences to generate for the demo"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=2, 
        help="Number of training epochs (keep small for demo)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--seq_type", 
        type=str, 
        default="dna", 
        choices=["dna", "rna"], 
        help="Type of sequences to generate (DNA or RNA)"
    )
    parser.add_argument(
        "--skip_generation", 
        action="store_true", 
        help="Skip data generation step if data already exists"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def generate_demo_data(args):
    """
    Generate synthetic data for the demo.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to the generated data file
    """
    logger.info("Generating synthetic dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_path = os.path.join(args.output_dir, f"{args.seq_type}_classification_data.csv")
    
    # Skip generation if requested and file exists
    if args.skip_generation and os.path.exists(data_path):
        logger.info(f"Using existing data file: {data_path}")
        return data_path
    
    # Run the data generation script
    cmd = [
        "python", "-m", "data.generate_sample_data",
        "--output_dir", args.output_dir,
        "--num_sequences", str(args.num_sequences),
        "--seq_length", "100",
        "--num_classes", str(args.num_classes),
        "--seq_type", args.seq_type,
        "--motif_length", "8",
        "--seed", str(args.seed)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error generating data: {result.stderr}")
        sys.exit(1)
    
    logger.info(f"Data generated successfully: {data_path}")
    return data_path

def train_model(args, data_path):
    """
    Train a classifier model on the synthetic data.
    
    Args:
        args: Command line arguments
        data_path: Path to the training data file
        
    Returns:
        Tuple of (model, training history, label mapping)
    """
    logger.info("Training classifier model...")
    
    # Initialize Evo2 model
    logger.info(f"Loading Evo2 model: {args.model_name}")
    evo2_model = Evo2(model_name=args.model_name)
    
    # Load data
    df = pd.read_csv(data_path)
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()
    
    # Create train/validation split
    from sklearn.model_selection import train_test_split
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    
    logger.info(f"Training set size: {len(train_seqs)}")
    logger.info(f"Validation set size: {len(val_seqs)}")
    
    # Create datasets
    train_dataset = SequenceClassificationDataset(
        sequences=train_seqs,
        labels=train_labels,
        tokenizer=evo2_model.tokenizer,
        max_length=256
    )
    
    val_dataset = SequenceClassificationDataset(
        sequences=val_seqs,
        labels=val_labels,
        tokenizer=evo2_model.tokenizer,
        max_length=256
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize classifier model
    model = Evo2Classifier(
        evo2_model=evo2_model,
        num_classes=args.num_classes,
        dropout_rate=0.1,
        freeze_base_model=True,
        pooling_strategy="mean"
    )
    
    # Initialize trainer
    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    trainer = Evo2FineTuner(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=5e-5,
        output_dir=model_dir
    )
    
    # Train model
    history = trainer.train(num_epochs=args.num_epochs)
    
    # Save final model
    trainer.save_checkpoint(args.num_epochs)
    
    # Create label mapping (for this demo it's just the index)
    label_map = {i: str(i) for i in range(args.num_classes)}
    
    return model, history, label_map

def evaluate_model(model, args, data_path):
    """
    Evaluate the trained model and visualize results.
    
    Args:
        model: Trained Evo2Classifier model
        args: Command line arguments
        data_path: Path to the data file
    """
    logger.info("Evaluating model and generating visualizations...")
    
    # Initialize Evo2 model for tokenization
    evo2_model = Evo2(model_name=args.model_name)
    
    # Load data
    df = pd.read_csv(data_path)
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()
    
    # Create dataset
    dataset = SequenceClassificationDataset(
        sequences=sequences,
        labels=labels,
        tokenizer=evo2_model.tokenizer,
        max_length=256
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Make predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"]
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Print classification report
    class_names = [f"Class {i}" for i in range(args.num_classes)]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    
    # Save classification report to file
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Plot class probabilities distribution
    plt.figure(figsize=(12, 8))
    
    for i in range(args.num_classes):
        # Get probabilities for this class
        class_probs = [probs[j] for j, label in enumerate(all_labels) if label == i]
        class_probs = np.array(class_probs)
        
        plt.subplot(1, args.num_classes, i+1)
        plt.hist(class_probs[:, i], bins=20, alpha=0.7, label=f'True Class {i}')
        
        # If binary classification, also plot probabilities for the other class
        if args.num_classes == 2:
            other_class = 1 - i
            plt.hist(class_probs[:, other_class], bins=20, alpha=0.5, label=f'Class {other_class}')
        
        plt.xlabel(f'Probability of Class {i}')
        plt.ylabel('Count')
        plt.title(f'Class {i} Probability Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "probability_distribution.png"))
    plt.close()

def make_predictions(model, args):
    """
    Make predictions on new data.
    
    Args:
        model: Trained Evo2Classifier model
        args: Command line arguments
    """
    logger.info("Making predictions on new sequences...")
    
    # Initialize Evo2 model for tokenization
    evo2_model = Evo2(model_name=args.model_name)
    
    # Create some new sequences based on sequence type
    if args.seq_type == "dna":
        new_sequences = [
            "ACGTACGTACGTACGTACGT",
            "TGCATGCATGCATGCATGCA",
            "GCTAGCTAGCTAGCTAGCTA",
            "ATGCATGCATGCATGCATGC"
        ]
    else:  # RNA
        new_sequences = [
            "ACGUACGUACGUACGUACGU",
            "UGCAUGCAUGCAUGCAUGCA",
            "GCUAGCUAGCUAGCUAGCUA",
            "AUGCAUGCAUGCAUGCAUGC"
        ]
    
    # Create dummy labels (not used for prediction)
    dummy_labels = [0] * len(new_sequences)
    
    # Create dataset
    dataset = SequenceClassificationDataset(
        sequences=new_sequences,
        labels=dummy_labels,
        tokenizer=evo2_model.tokenizer,
        max_length=256
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Make predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Create results DataFrame
    results = []
    for i, (seq, pred) in enumerate(zip(new_sequences, all_preds)):
        probs = all_probs[i]
        
        result = {
            "sequence": seq,
            "prediction": pred,
        }
        
        # Add probabilities
        for j, prob in enumerate(probs):
            result[f"prob_class_{j}"] = prob
            
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Predictions saved to {results_path}")
    logger.info("\nPrediction Results:")
    for i, row in results_df.iterrows():
        pred_class = row["prediction"]
        pred_prob = row[f"prob_class_{pred_class}"]
        logger.info(f"Sequence {i+1}: Class {pred_class} (probability: {pred_prob:.4f})")

def main():
    """Main function for the demo."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Generate demo data
        data_path = generate_demo_data(args)
        
        # Train model
        model, history, label_map = train_model(args, data_path)
        
        # Evaluate model
        evaluate_model(model, args, data_path)
        
        # Make predictions on new data
        make_predictions(model, args)
        
        logger.info(f"Demo completed successfully! Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 