#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction script for using trained Evo2 classifier models.
This script handles loading a trained model and making predictions on new data.
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm

from classifier import (
    Evo2Classifier, 
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
    parser = argparse.ArgumentParser(description="Make predictions with a trained Evo2 classifier")
    
    # Data arguments
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to CSV file with sequences to predict"
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
        default=None, 
        help="Column name for ground truth labels (optional)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="predictions.csv", 
        help="Path to save prediction results"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="evo2_7b", 
        choices=["evo2_1b_base", "evo2_7b", "evo2_7b_base", "evo2_40b", "evo2_40b_base"],
        help="Base Evo2 model name"
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=2, 
        help="Number of output classes"
    )
    parser.add_argument(
        "--label_map_path", 
        type=str, 
        default=None, 
        help="Path to JSON file with label mapping"
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
    
    # Other arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to use for inference"
    )
    parser.add_argument(
        "--include_probabilities", 
        action="store_true", 
        help="Include prediction probabilities in output"
    )
    
    return parser.parse_args()

def load_data(data_path: str, seq_col: str, label_col=None):
    """
    Load data for prediction.
    
    Args:
        data_path: Path to the CSV or TSV file
        seq_col: Column name for sequences
        label_col: Column name for ground truth labels (optional)
        
    Returns:
        Tuple of (sequences, labels)
    """
    # Load data
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".tsv"):
        df = pd.read_csv(data_path, sep="\t")
    else:
        raise ValueError("Unsupported file format. Expected .csv or .tsv")
    
    # Extract sequences
    sequences = df[seq_col].tolist()
    
    # Extract labels if provided
    labels = None
    if label_col is not None and label_col in df.columns:
        labels = df[label_col].tolist()
    
    return sequences, labels

def load_model(args):
    """
    Initialize and load the trained model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Loaded model and Evo2 base model
    """
    # Initialize Evo2 model
    evo2_model = Evo2(model_name=args.model_name)
    
    # Initialize classifier model
    model = Evo2Classifier(
        evo2_model=evo2_model,
        num_classes=args.num_classes,
        pooling_strategy=args.pooling_strategy,
        embedding_layer=args.embedding_layer
    )
    
    # Load model checkpoint
    logger.info(f"Loading model checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(args.device)
    model.eval()
    
    return model, evo2_model

def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    sequences, labels = load_data(args.data_path, args.seq_col, args.label_col)
    
    logger.info(f"Loaded {len(sequences)} sequences for prediction")
    
    # Load label mapping if provided
    label_map = None
    inverse_label_map = None
    if args.label_map_path is not None:
        with open(args.label_map_path, "r") as f:
            label_map = json.load(f)
            # Create inverse mapping (index -> label)
            inverse_label_map = {int(v): k for k, v in label_map.items()}
    
    # Load trained model
    model, evo2_base = load_model(args)
    
    # Create dataset
    if labels is not None:
        dataset = SequenceClassificationDataset(
            sequences=sequences,
            labels=labels,
            tokenizer=evo2_base.tokenizer,
            max_length=args.max_length
        )
    else:
        # Create dummy labels
        dummy_labels = [0] * len(sequences)
        dataset = SequenceClassificationDataset(
            sequences=sequences,
            labels=dummy_labels,
            tokenizer=evo2_base.tokenizer,
            max_length=args.max_length
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Make predictions
    logger.info("Making predictions")
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move batch to device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Prepare output
    results = []
    for i, (sequence, pred) in enumerate(zip(sequences, all_preds)):
        # Get class probabilities
        probs = all_probs[i]
        
        # Convert prediction to original label if mapping is available
        if inverse_label_map is not None:
            pred_label = inverse_label_map.get(pred, pred)
        else:
            pred_label = pred
        
        result = {
            "sequence": sequence,
            "prediction": pred_label
        }
        
        # Add ground truth if available
        if labels is not None:
            result["true_label"] = labels[i]
        
        # Add probabilities if requested
        if args.include_probabilities:
            for j, prob in enumerate(probs):
                class_name = inverse_label_map.get(j, j) if inverse_label_map else j
                result[f"prob_{class_name}"] = prob
        
        results.append(result)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    
    logger.info(f"Predictions saved to {args.output_path}")
    
    # Calculate metrics if ground truth is available
    if labels is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Convert labels to integers if they are strings and mapping is available
        if label_map is not None and isinstance(labels[0], str):
            labels = [label_map[label] for label in labels]
        
        # Calculate metrics
        accuracy = accuracy_score(labels, all_preds)
        precision = precision_score(labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(labels, all_preds, average="macro", zero_division=0)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main() 