#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classifier module for fine-tuning Evo2 models on sequence classification tasks.
This module provides classes for creating, training, and evaluating Evo2-based classifiers.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SequenceClassificationDataset(Dataset):
    """
    Dataset for sequence classification tasks.
    
    Attributes:
        sequences (List[str]): List of DNA/RNA sequences
        labels (torch.Tensor): Tensor of labels for each sequence
        tokenizer: Tokenizer to convert sequences to token IDs
        max_length (int): Maximum sequence length
    """
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of input sequences
            labels: List of corresponding labels
            tokenizer: Tokenizer to convert sequences to token IDs
            max_length: Maximum sequence length (will be truncated if longer)
        """
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "label": label
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader that handles variable sequence lengths.
    
    Args:
        batch: List of items from the dataset
        
    Returns:
        Dictionary with batched input_ids, attention_mask, and labels
    """
    # Get max sequence length in this batch
    max_length = max([item["input_ids"].size(0) for item in batch])
    
    # Prepare tensors
    batch_size = len(batch)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = 1
        labels[i] = item["label"]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class Evo2Classifier(nn.Module):
    """
    Fine-tuned Evo2 model for sequence classification.
    
    This model adds a classification head on top of the Evo2 model to perform
    classification tasks on DNA/RNA sequences.
    """
    
    def __init__(
        self, 
        evo2_model,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_base_model: bool = True,
        pooling_strategy: str = "mean",
        embedding_layer: str = "model.blocks.49"
    ):
        """
        Initialize the classifier.
        
        Args:
            evo2_model: Initialized Evo2 model instance
            num_classes: Number of output classes
            dropout_rate: Dropout probability for the classification head
            freeze_base_model: Whether to freeze the base Evo2 model parameters
            pooling_strategy: How to pool sequence embeddings ("mean", "max", or "cls")
            embedding_layer: Which layer to extract embeddings from
        """
        super(Evo2Classifier, self).__init__()
        
        # Store the base model
        self.evo2_model = evo2_model
        
        # Get hidden size from model config
        hidden_size = self.evo2_model.model.config.hidden_size
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.evo2_model.model.parameters():
                param.requires_grad = False
                
        self.pooling_strategy = pooling_strategy
        self.embedding_layer = embedding_layer
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token IDs
            attention_mask: Attention mask for padding (optional)
            
        Returns:
            Logits for each class
        """
        # Get embeddings from the specified layer
        _, embeddings = self.evo2_model.forward(
            input_ids, 
            return_embeddings=True,
            layer_names=[self.embedding_layer]
        )
        
        # Extract embeddings from the specified layer
        sequence_embeddings = embeddings[self.embedding_layer]
        
        # Apply pooling strategy
        if self.pooling_strategy == "mean":
            # Mean pooling (considering attention mask if provided)
            if attention_mask is not None:
                # Expand mask to match embedding dimensions
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_embeddings.size())
                # Apply mask and calculate mean
                pooled = (sequence_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = sequence_embeddings.mean(dim=1)
                
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                # Create a mask for padded values (set to large negative value)
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_embeddings.size())
                masked_embeddings = sequence_embeddings.clone()
                masked_embeddings[mask_expanded == 0] = -1e9
                pooled = torch.max(masked_embeddings, dim=1)[0]
            else:
                pooled = torch.max(sequence_embeddings, dim=1)[0]
                
        elif self.pooling_strategy == "cls":
            # Use the first token's embedding as the sequence representation
            pooled = sequence_embeddings[:, 0, :]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply classifier head
        logits = self.classifier(pooled)
        
        return logits

class Evo2FineTuner:
    """
    Trainer class for fine-tuning Evo2 models for classification tasks.
    """
    
    def __init__(
        self,
        model: Evo2Classifier,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./evo2_classifier_output"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Evo2Classifier model
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to use for training ("cuda" or "cpu")
            output_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = output_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }
    
    def train(self, num_epochs: int, evaluate_every: int = 1, save_every: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            evaluate_every: Evaluate model every N epochs
            save_every: Save model checkpoint every N epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Calculate average training loss
            train_loss /= len(self.train_dataloader)
            self.history["train_loss"].append(train_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Evaluate if needed
            if self.val_dataloader is not None and (epoch + 1) % evaluate_every == 0:
                val_metrics = self.evaluate()
                
                # Log validation metrics
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"Accuracy: {val_metrics['accuracy']:.4f}, "
                            f"F1: {val_metrics['f1']:.4f}")
            
            # Save checkpoint if needed
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Update metrics
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Add to lists for metric calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(self.val_dataloader)
        
        # Calculate validation metrics
        metrics = {
            "loss": val_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
        }
        
        # Update history
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(metrics["accuracy"])
        self.history["val_precision"].append(metrics["precision"])
        self.history["val_recall"].append(metrics["recall"])
        self.history["val_f1"].append(metrics["f1"])
        
        return metrics
    
    def save_checkpoint(self, epoch: int) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load history
        self.history = checkpoint["history"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"]
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for the given dataloader.
        
        Args:
            dataloader: DataLoader with test data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        return np.array(all_preds), np.array(all_probs) 