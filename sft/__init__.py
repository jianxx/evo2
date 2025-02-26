"""
Evo2 Sequence Fine-Tuning Framework

A framework for fine-tuning Evo2 models on sequence classification tasks,
particularly DNA/RNA sequence analysis.
"""

from classifier import (
    SequenceClassificationDataset,
    collate_fn,
    Evo2Classifier,
    Evo2FineTuner
)

__version__ = "0.1.0" 