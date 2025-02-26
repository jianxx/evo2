#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample data generation script for Evo2 fine-tuning.
This script generates synthetic DNA/RNA sequences with class-specific patterns for testing classification models.
"""

import os
import random
import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple, Dict

# Define nucleotides and their complements for DNA
DNA_NUCLEOTIDES = ['A', 'C', 'G', 'T']
DNA_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# Define nucleotides for RNA
RNA_NUCLEOTIDES = ['A', 'C', 'G', 'U']
RNA_COMPLEMENT = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate sample sequence data for classification")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./", 
        help="Directory to save generated data"
    )
    parser.add_argument(
        "--num_sequences", 
        type=int, 
        default=1000, 
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--seq_length", 
        type=int, 
        default=200, 
        help="Length of each sequence"
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=3, 
        help="Number of classes to generate"
    )
    parser.add_argument(
        "--seq_type", 
        type=str, 
        default="dna", 
        choices=["dna", "rna"], 
        help="Type of sequence (DNA or RNA)"
    )
    parser.add_argument(
        "--motif_length", 
        type=int, 
        default=8, 
        help="Length of motifs to insert"
    )
    parser.add_argument(
        "--motifs_per_class", 
        type=int, 
        default=3, 
        help="Number of motifs per class"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def generate_random_sequence(length: int, nucleotides: List[str]) -> str:
    """
    Generate a random DNA or RNA sequence.
    
    Args:
        length: Length of the sequence to generate
        nucleotides: List of nucleotides to use
        
    Returns:
        Random sequence
    """
    return ''.join(random.choice(nucleotides) for _ in range(length))

def generate_motif(length: int, nucleotides: List[str]) -> str:
    """
    Generate a random motif.
    
    Args:
        length: Length of the motif
        nucleotides: List of nucleotides to use
        
    Returns:
        Generated motif
    """
    return ''.join(random.choice(nucleotides) for _ in range(length))

def insert_motif(sequence: str, motif: str, position: int = None) -> str:
    """
    Insert a motif into a sequence at a specific position.
    
    Args:
        sequence: The original sequence
        motif: The motif to insert
        position: Position to insert the motif (random if None)
        
    Returns:
        Modified sequence with the inserted motif
    """
    if position is None:
        # Random position that ensures the whole motif fits in the sequence
        position = random.randint(0, len(sequence) - len(motif))
    
    return sequence[:position] + motif + sequence[position + len(motif):]

def reverse_complement(sequence: str, complement_map: Dict[str, str]) -> str:
    """
    Get the reverse complement of a DNA or RNA sequence.
    
    Args:
        sequence: The original sequence
        complement_map: Dictionary mapping nucleotides to their complements
        
    Returns:
        Reverse complement sequence
    """
    return ''.join(complement_map.get(nucleotide, nucleotide) for nucleotide in reversed(sequence))

def generate_class_sequences(
    num_sequences: int, 
    seq_length: int, 
    motifs: List[str], 
    nucleotides: List[str],
    complement_map: Dict[str, str],
    motif_probability: float = 0.8,
    class_id: int = 0
) -> List[Tuple[str, int]]:
    """
    Generate sequences for a specific class.
    
    Args:
        num_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        motifs: List of motifs to insert into sequences of this class
        nucleotides: List of nucleotides to use
        complement_map: Dictionary mapping nucleotides to their complements
        motif_probability: Probability of inserting a motif
        class_id: Class identifier
        
    Returns:
        List of (sequence, class_id) tuples
    """
    sequences = []
    
    for _ in range(num_sequences):
        # Generate a random sequence
        sequence = generate_random_sequence(seq_length, nucleotides)
        
        # Insert motifs with some probability
        if random.random() < motif_probability:
            # Choose a random motif from the list
            motif = random.choice(motifs)
            
            # Randomly decide whether to use the original motif or its reverse complement
            if random.random() < 0.5:
                motif = reverse_complement(motif, complement_map)
                
            # Insert the motif
            sequence = insert_motif(sequence, motif)
        
        sequences.append((sequence, class_id))
    
    return sequences

def generate_dataset(
    num_sequences: int, 
    seq_length: int, 
    num_classes: int,
    seq_type: str = "dna",
    motif_length: int = 8,
    motifs_per_class: int = 3,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], Dict[int, List[str]]]:
    """
    Generate a dataset of DNA or RNA sequences with class labels.
    
    Args:
        num_sequences: Total number of sequences to generate
        seq_length: Length of each sequence
        num_classes: Number of classes
        seq_type: Type of sequence ("dna" or "rna")
        motif_length: Length of motifs to generate
        motifs_per_class: Number of motifs per class
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dataset, class_motifs)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Determine nucleotides and complements based on sequence type
    if seq_type.lower() == "dna":
        nucleotides = DNA_NUCLEOTIDES
        complement_map = DNA_COMPLEMENT
    else:
        nucleotides = RNA_NUCLEOTIDES
        complement_map = RNA_COMPLEMENT
    
    # Define class motifs (each class has its own set of motifs)
    class_motifs = {}
    for class_id in range(num_classes):
        # Generate motifs per class
        motifs = [generate_motif(motif_length, nucleotides) for _ in range(motifs_per_class)]
        class_motifs[class_id] = motifs
    
    # Calculate sequences per class (approximately equal distribution)
    seqs_per_class = num_sequences // num_classes
    remainder = num_sequences % num_classes
    
    # Generate sequences for each class
    all_sequences = []
    for class_id in range(num_classes):
        # Add an extra sequence to some classes if there's a remainder
        class_size = seqs_per_class + (1 if class_id < remainder else 0)
        
        class_sequences = generate_class_sequences(
            num_sequences=class_size,
            seq_length=seq_length,
            motifs=class_motifs[class_id],
            nucleotides=nucleotides,
            complement_map=complement_map,
            class_id=class_id
        )
        
        all_sequences.extend(class_sequences)
    
    # Shuffle the dataset
    random.shuffle(all_sequences)
    
    return all_sequences, class_motifs

def main():
    """Main function to generate the dataset."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    print(f"Generating {args.num_sequences} {args.seq_type.upper()} sequences with {args.num_classes} classes...")
    dataset, class_motifs = generate_dataset(
        num_sequences=args.num_sequences,
        seq_length=args.seq_length,
        num_classes=args.num_classes,
        seq_type=args.seq_type,
        motif_length=args.motif_length,
        motifs_per_class=args.motifs_per_class,
        seed=args.seed
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset, columns=["sequence", "label"])
    
    # Save to CSV
    output_path = os.path.join(args.output_dir, f"{args.seq_type}_classification_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
    # Save class motifs for reference
    class_motifs_df = pd.DataFrame({
        'class': [class_id for class_id, motifs in class_motifs.items() for _ in motifs],
        'motif': [motif for motifs in class_motifs.values() for motif in motifs]
    })
    
    motifs_path = os.path.join(args.output_dir, f"{args.seq_type}_class_motifs.csv")
    class_motifs_df.to_csv(motifs_path, index=False)
    print(f"Class motifs saved to {motifs_path}")
    
    # Print class distribution
    class_counts = df['label'].value_counts().sort_index()
    print("\nClass distribution:")
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} sequences")
    
    # Print motif information
    print("\nClass motifs:")
    for class_id, motifs in class_motifs.items():
        print(f"Class {class_id}:")
        for i, motif in enumerate(motifs):
            print(f"  Motif {i+1}: {motif} (length: {len(motif)})")

if __name__ == "__main__":
    main() 