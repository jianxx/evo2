#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Evo2 classifier module.
"""

import unittest
import torch
import numpy as np
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import (
    SequenceClassificationDataset,
    collate_fn,
    Evo2Classifier,
    Evo2FineTuner
)

class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def tokenize(self, sequence):
        """Mock tokenize method."""
        # Simply convert each character to its ASCII value as a token
        return [ord(c) for c in sequence]

class MockEvo2:
    """Mock Evo2 model for testing purposes."""
    
    def __init__(self):
        """Initialize mock model."""
        # Mock config with hidden size
        self.model = MagicMock()
        self.model.config = MagicMock()
        self.model.config.hidden_size = 768
        self.tokenizer = MockTokenizer()
    
    def forward(self, input_ids, return_embeddings=False, layer_names=None):
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        hidden_size = self.model.config.hidden_size
        
        # Create fake embeddings
        embeddings = {}
        for layer_name in layer_names:
            embeddings[layer_name] = torch.randn(batch_size, seq_len, hidden_size)
        
        return None, embeddings

class TestSequenceClassificationDataset(unittest.TestCase):
    """Test cases for SequenceClassificationDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sequences = ["ACGT", "TGCA", "GCTA"]
        self.labels = [0, 1, 0]
        self.tokenizer = MockTokenizer()
        self.max_length = 10
        self.dataset = SequenceClassificationDataset(
            sequences=self.sequences,
            labels=self.labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.dataset), 3)
    
    def test_getitem(self):
        """Test __getitem__ method."""
        item = self.dataset[0]
        
        # Check returned item structure
        self.assertIn("input_ids", item)
        self.assertIn("label", item)
        
        # Check item values
        expected_tokens = torch.tensor([ord(c) for c in "ACGT"], dtype=torch.long)
        self.assertTrue(torch.all(item["input_ids"].eq(expected_tokens)))
        self.assertEqual(item["label"].item(), 0)
    
    def test_truncation(self):
        """Test sequence truncation."""
        # Create a long sequence
        long_sequence = "ACGT" * 10  # 40 characters, greater than max_length
        
        # Override dataset with the long sequence
        dataset = SequenceClassificationDataset(
            sequences=[long_sequence],
            labels=[0],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Get item
        item = dataset[0]
        
        # Check truncation
        self.assertEqual(len(item["input_ids"]), self.max_length)

class TestCollateFn(unittest.TestCase):
    """Test cases for collate_fn."""
    
    def test_collate_fn(self):
        """Test collate_fn functionality."""
        # Create batch items
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "label": torch.tensor(0)},
            {"input_ids": torch.tensor([4, 5]), "label": torch.tensor(1)},
            {"input_ids": torch.tensor([6, 7, 8, 9]), "label": torch.tensor(2)}
        ]
        
        # Apply collate_fn
        collated = collate_fn(batch)
        
        # Check collated batch structure
        self.assertIn("input_ids", collated)
        self.assertIn("attention_mask", collated)
        self.assertIn("labels", collated)
        
        # Check batch sizes
        self.assertEqual(collated["input_ids"].shape[0], 3)
        self.assertEqual(collated["attention_mask"].shape[0], 3)
        self.assertEqual(collated["labels"].shape[0], 3)
        
        # Check sequence padding
        self.assertEqual(collated["input_ids"].shape[1], 4)  # Max sequence length
        self.assertEqual(collated["attention_mask"].shape[1], 4)
        
        # Check padding and attention mask
        self.assertEqual(collated["input_ids"][0, 3].item(), 0)  # Padding token
        self.assertEqual(collated["attention_mask"][0, 0].item(), 1)  # Attention on real token
        self.assertEqual(collated["attention_mask"][0, 3].item(), 0)  # No attention on padding

class TestEvo2Classifier(unittest.TestCase):
    """Test cases for Evo2Classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evo2_model = MockEvo2()
        self.num_classes = 3
        self.model = Evo2Classifier(
            evo2_model=self.evo2_model,
            num_classes=self.num_classes,
            dropout_rate=0.1,
            freeze_base_model=True,
            pooling_strategy="mean",
            embedding_layer="model.blocks.49"
        )
    
    def test_forward_mean_pooling(self):
        """Test forward pass with mean pooling."""
        # Create input batch
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Call forward
        logits = self.model(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, self.num_classes))
    
    def test_forward_max_pooling(self):
        """Test forward pass with max pooling."""
        # Create model with max pooling
        model = Evo2Classifier(
            evo2_model=self.evo2_model,
            num_classes=self.num_classes,
            pooling_strategy="max"
        )
        
        # Create input batch
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Call forward
        logits = model(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, self.num_classes))
    
    def test_forward_cls_pooling(self):
        """Test forward pass with cls pooling."""
        # Create model with cls pooling
        model = Evo2Classifier(
            evo2_model=self.evo2_model,
            num_classes=self.num_classes,
            pooling_strategy="cls"
        )
        
        # Create input batch
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        # Call forward
        logits = model(input_ids)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, self.num_classes))
    
    def test_invalid_pooling_strategy(self):
        """Test that invalid pooling strategy raises error."""
        with self.assertRaises(ValueError):
            model = Evo2Classifier(
                evo2_model=self.evo2_model,
                pooling_strategy="invalid"
            )
            
            batch_size = 2
            seq_len = 5
            input_ids = torch.randint(0, 100, (batch_size, seq_len))
            model(input_ids)

class TestEvo2FineTuner(unittest.TestCase):
    """Test cases for Evo2FineTuner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evo2_model = MockEvo2()
        self.model = Evo2Classifier(
            evo2_model=self.evo2_model,
            num_classes=2
        )
        
        # Create mock dataloaders
        self.train_dataloader = MagicMock()
        self.train_dataloader.__len__ = MagicMock(return_value=10)
        self.train_dataloader.__iter__ = MagicMock(return_value=iter([
            {
                "input_ids": torch.randint(0, 100, (2, 5)),
                "attention_mask": torch.ones(2, 5),
                "labels": torch.tensor([0, 1])
            }
            for _ in range(10)
        ]))
        
        self.val_dataloader = MagicMock()
        self.val_dataloader.__len__ = MagicMock(return_value=5)
        self.val_dataloader.__iter__ = MagicMock(return_value=iter([
            {
                "input_ids": torch.randint(0, 100, (2, 5)),
                "attention_mask": torch.ones(2, 5),
                "labels": torch.tensor([0, 1])
            }
            for _ in range(5)
        ]))
        
        # Create trainer
        self.output_dir = tempfile.mkdtemp()
        self.trainer = Evo2FineTuner(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            output_dir=self.output_dir,
            device="cpu"
        )
    
    def test_train(self):
        """Test training functionality."""
        # Run training for 1 epoch
        history = self.trainer.train(num_epochs=1)
        
        # Check history contains expected keys
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("val_accuracy", history)
        self.assertIn("val_f1", history)
        
        # Check history has correct lengths
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertEqual(len(history["val_loss"]), 1)
    
    def test_evaluate(self):
        """Test evaluation functionality."""
        # Run evaluation
        metrics = self.trainer.evaluate()
        
        # Check metrics contains expected keys
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        # Train for 1 epoch to initialize history
        self.trainer.train(num_epochs=1)
        
        # Save checkpoint
        checkpoint_path = self.trainer.save_checkpoint(1)
        
        # Check file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Create new trainer
        new_trainer = Evo2FineTuner(
            model=Evo2Classifier(evo2_model=MockEvo2(), num_classes=2),
            train_dataloader=self.train_dataloader,
            output_dir=self.output_dir,
            device="cpu"
        )
        
        # Load checkpoint
        epoch = new_trainer.load_checkpoint(checkpoint_path)
        
        # Check loaded epoch
        self.assertEqual(epoch, 1)
        
        # Check history was loaded
        self.assertIn("train_loss", new_trainer.history)
        self.assertEqual(len(new_trainer.history["train_loss"]), 1)
    
    def tearDown(self):
        """Clean up temporary files."""
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)

if __name__ == "__main__":
    unittest.main() 