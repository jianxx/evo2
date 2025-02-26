#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for fine-tuning Evo2 models on sequence classification tasks.
This script handles data loading, model initialization, training, and evaluation.
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import logging

from classifier import (
    Evo2Classifier, 
    Evo2FineTuner, 
    SequenceClassificationDataset,
    collate_fn,
    logger
)

# Import Evo2 model - assuming it's available in the environment
try:
    from evo2 import Evo2
except ImportError:
    logger.error("Could not import Evo2. Make sure it's installed.")
    raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Evo2 classifier model")
    
    # Data arguments
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to CSV file with sequences and labels"
    )
    parser.add_argument(
        "--seq_col", 
        type=str, 
        default="sequence", 
        help="Column name for sequences"
    )
    parser.add_argument(
        "--label_col", 
        type=str, 
        default="label", 
        help="Column name for labels"
    )
    parser.add_argument(
        "--val_size", 
        type=float, 
        default=0.2, 
        help="Validation set size (proportion of training data)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum sequence length"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="evo2_7b", 
        choices=["evo2_1b_base", "evo2_7b", "evo2_7b_base", "evo2_40b", "evo2_40b_base"],
        help="Evo2 model to use"
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=2, 
        help="Number of output classes (use -1 to determine automatically)"
    )
    parser.add_argument(
        "--dropout_rate", 
        type=float, 
        default=0.1, 
        help="Dropout rate for the classification head"
    )
    parser.add_argument(
        "--freeze_base_model", 
        action="store_true", 
        help="Freeze parameters of the base Evo2 model"
    )
    parser.add_argument(
        "--pooling_strategy", 
        type=str, 
        default="mean", 
        choices=["mean", "max", "cls"], 
        help="Strategy for pooling sequence embeddings"
    )
    parser.add_argument(
        "--embedding_layer", 
        type=str, 
        default="model.blocks.49",
        help="Layer to extract embeddings from"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=5, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evo2_classifier_output", 
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--evaluate_every", 
        type=int, 
        default=1, 
        help="Evaluate model every N epochs"
    )
    parser.add_argument(
        "--save_every", 
        type=int, 
        default=1, 
        help="Save model checkpoint every N epochs"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def load_data(data_path: str, seq_col: str, label_col: str):
    """
    Load and preprocess the dataset.
    
    Args:
        data_path: Path to the CSV or TSV file
        seq_col: Column name for sequences
        label_col: Column name for labels
        
    Returns:
        Tuple of (sequences, labels, num_classes, label_map)
    """
    # Load data
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".tsv"):
        df = pd.read_csv(data_path, sep="\t")
    else:
        raise ValueError("Unsupported file format. Expected .csv or .tsv")
    
    # Extract sequences and labels
    sequences = df[seq_col].tolist()
    
    # Convert labels to integers if they are strings
    label_map = None
    if df[label_col].dtype == "object":
        # Create a mapping from label strings to integers
        unique_labels = sorted(df[label_col].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert labels to integers
        labels = [label_map[label] for label in df[label_col]]
        
        logger.info(f"Label mapping: {label_map}")
    else:
        labels = df[label_col].tolist()
    
    return sequences, labels, len(set(labels)), label_map

def plot_training_history(history, output_dir):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training metrics
        output_dir: Directory to save plots
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
    
    # Plot validation metrics
    if "val_accuracy" in history and history["val_accuracy"]:
        plt.figure(figsize=(10, 5))
        plt.plot(history["val_accuracy"], label="Accuracy")
        plt.plot(history["val_f1"], label="F1 Score")
        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "metrics_plot.png"))
        plt.close()

def main():
    """Main function for training the classifier."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    sequences, labels, num_classes, label_map = load_data(args.data_path, args.seq_col, args.label_col)
    
    # Determine number of classes if not specified
    if args.num_classes == -1:
        args.num_classes = num_classes
        logger.info(f"Number of classes automatically set to {num_classes}")
    
    # Split data into train and validation sets
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=args.val_size, random_state=args.seed, stratify=labels
    )
    
    logger.info(f"Training set size: {len(train_sequences)}")
    logger.info(f"Validation set size: {len(val_sequences)}")
    
    # Save label mapping if available
    if label_map:
        with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)
    
    # Initialize Evo2 model and tokenizer
    logger.info(f"Loading Evo2 model: {args.model_name}")
    evo2_model = Evo2(model_name=args.model_name)
    
    # Create datasets
    train_dataset = SequenceClassificationDataset(
        sequences=train_sequences,
        labels=train_labels,
        tokenizer=evo2_model.tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = SequenceClassificationDataset(
        sequences=val_sequences,
        labels=val_labels,
        tokenizer=evo2_model.tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
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
        dropout_rate=args.dropout_rate,
        freeze_base_model=args.freeze_base_model,
        pooling_strategy=args.pooling_strategy,
        embedding_layer=args.embedding_layer
    )
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Evo2FineTuner(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )
    
    # Train model
    logger.info("Starting training")
    history = trainer.train(
        num_epochs=args.num_epochs,
        evaluate_every=args.evaluate_every,
        save_every=args.save_every
    )
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Save training arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        # Convert argparse Namespace to dictionary
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2)
    
    logger.info(f"Training complete. Model checkpoints saved to {args.output_dir}")

if __name__ == "__main__":
    main() 