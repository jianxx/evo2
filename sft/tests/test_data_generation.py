#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the sample data generation module.
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import random

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generate_sample_data import (
    generate_random_sequence,
    generate_motif,
    insert_motif,
    reverse_complement,
    generate_class_sequences,
    generate_dataset,
    DNA_NUCLEOTIDES,
    DNA_COMPLEMENT,
    RNA_NUCLEOTIDES,
    RNA_COMPLEMENT
)

class TestSequenceGeneration(unittest.TestCase):
    """Test cases for sequence generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed seed for reproducibility
        random.seed(42)
    
    def test_generate_random_sequence(self):
        """Test random sequence generation."""
        # Test DNA sequence generation
        dna_seq = generate_random_sequence(10, DNA_NUCLEOTIDES)
        
        # Check length
        self.assertEqual(len(dna_seq), 10)
        
        # Check valid nucleotides
        for nucleotide in dna_seq:
            self.assertIn(nucleotide, DNA_NUCLEOTIDES)
        
        # Test RNA sequence generation
        rna_seq = generate_random_sequence(10, RNA_NUCLEOTIDES)
        
        # Check length
        self.assertEqual(len(rna_seq), 10)
        
        # Check valid nucleotides
        for nucleotide in rna_seq:
            self.assertIn(nucleotide, RNA_NUCLEOTIDES)
    
    def test_generate_motif(self):
        """Test motif generation."""
        # Test DNA motif generation
        dna_motif = generate_motif(5, DNA_NUCLEOTIDES)
        
        # Check length
        self.assertEqual(len(dna_motif), 5)
        
        # Check valid nucleotides
        for nucleotide in dna_motif:
            self.assertIn(nucleotide, DNA_NUCLEOTIDES)
        
        # Test RNA motif generation
        rna_motif = generate_motif(5, RNA_NUCLEOTIDES)
        
        # Check length
        self.assertEqual(len(rna_motif), 5)
        
        # Check valid nucleotides
        for nucleotide in rna_motif:
            self.assertIn(nucleotide, RNA_NUCLEOTIDES)
    
    def test_insert_motif(self):
        """Test motif insertion."""
        sequence = "ACGTACGT"
        motif = "GGG"
        
        # Test insertion at specific position
        position = 2
        modified_seq = insert_motif(sequence, motif, position)
        
        # Check motif was inserted at the correct position
        self.assertEqual(modified_seq[position:position+len(motif)], motif)
        
        # Check sequence length
        self.assertEqual(len(modified_seq), len(sequence))
        
        # Test insertion at random position
        random_modified_seq = insert_motif(sequence, motif)
        
        # Check sequence length
        self.assertEqual(len(random_modified_seq), len(sequence))
        
        # Check motif is present somewhere in the sequence
        self.assertTrue(motif in random_modified_seq)
    
    def test_reverse_complement(self):
        """Test reverse complement function."""
        # Test DNA reverse complement
        dna_seq = "ACGTACGT"
        dna_rc = reverse_complement(dna_seq, DNA_COMPLEMENT)
        
        # Check length
        self.assertEqual(len(dna_rc), len(dna_seq))
        
        # Check specific reverse complement
        self.assertEqual(dna_rc, "ACGTACGT"[::-1].translate(str.maketrans("ACGT", "TGCA")))
        
        # Test RNA reverse complement
        rna_seq = "ACGUACGU"
        rna_rc = reverse_complement(rna_seq, RNA_COMPLEMENT)
        
        # Check length
        self.assertEqual(len(rna_rc), len(rna_seq))
        
        # Check specific reverse complement
        self.assertEqual(rna_rc, "ACGUACGU"[::-1].translate(str.maketrans("ACGU", "UGCA")))

class TestDatasetGeneration(unittest.TestCase):
    """Test cases for dataset generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed seed for reproducibility
        random.seed(42)
    
    def test_generate_class_sequences(self):
        """Test class-specific sequence generation."""
        motifs = ["ATAT", "GCGC"]
        num_sequences = 5
        seq_length = 20
        
        # Generate sequences for a class
        sequences = generate_class_sequences(
            num_sequences=num_sequences,
            seq_length=seq_length,
            motifs=motifs,
            nucleotides=DNA_NUCLEOTIDES,
            complement_map=DNA_COMPLEMENT,
            class_id=1
        )
        
        # Check number of sequences
        self.assertEqual(len(sequences), num_sequences)
        
        # Check sequence length
        self.assertEqual(len(sequences[0][0]), seq_length)
        
        # Check class ID
        self.assertEqual(sequences[0][1], 1)
        
        # Check if at least one sequence contains a motif
        found_motif = False
        for seq, _ in sequences:
            if motifs[0] in seq or motifs[1] in seq:
                found_motif = True
                break
        
        self.assertTrue(found_motif)
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        num_sequences = 20
        seq_length = 15
        num_classes = 2
        
        # Generate DNA dataset
        dna_dataset, dna_motifs = generate_dataset(
            num_sequences=num_sequences,
            seq_length=seq_length,
            num_classes=num_classes,
            seq_type="dna",
            motif_length=4,
            motifs_per_class=2,
            seed=42
        )
        
        # Check dataset size
        self.assertEqual(len(dna_dataset), num_sequences)
        
        # Check class distribution (approximately equal)
        class_counts = {}
        for _, class_id in dna_dataset:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        for class_id in range(num_classes):
            self.assertIn(class_id, class_counts)
            # Check class count is within 1 of expected (due to remainders)
            expected_count = num_sequences // num_classes
            self.assertTrue(abs(class_counts[class_id] - expected_count) <= 1)
        
        # Check motifs
        self.assertEqual(len(dna_motifs), num_classes)
        for class_id in range(num_classes):
            self.assertIn(class_id, dna_motifs)
            self.assertEqual(len(dna_motifs[class_id]), 2)  # 2 motifs per class
        
        # Generate RNA dataset
        rna_dataset, rna_motifs = generate_dataset(
            num_sequences=num_sequences,
            seq_length=seq_length,
            num_classes=num_classes,
            seq_type="rna",
            motif_length=4,
            motifs_per_class=2,
            seed=42
        )
        
        # Check dataset size
        self.assertEqual(len(rna_dataset), num_sequences)
        
        # Check for RNA-specific nucleotides
        for seq, _ in rna_dataset:
            for nucleotide in seq:
                self.assertIn(nucleotide, RNA_NUCLEOTIDES)

if __name__ == "__main__":
    unittest.main() 