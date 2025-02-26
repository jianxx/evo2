# Evo2 Sequence Fine-Tuning Framework

This directory contains a framework for fine-tuning Evo2 models for sequence classification tasks, particularly focusing on DNA/RNA sequence analysis.

## Overview

The framework provides a complete pipeline for:

1. Training Evo2-based classifiers on DNA/RNA sequences
2. Evaluating model performance
3. Making predictions on new sequences
4. Generating synthetic datasets for testing and development

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- tqdm
- Evo2 (the main model package)

## Directory Structure

```
sft/
├── classifier.py         # Core classifier implementation
├── train_classifier.py   # Training script
├── predict.py            # Prediction script
├── demo.py               # Demo script showcasing the workflow
├── data/                 # Data utilities
│   └── generate_sample_data.py # Script for generating synthetic data
└── tests/                # Unit tests
    ├── test_classifier.py       # Tests for classifier module
    └── test_data_generation.py  # Tests for data generation
```

## Usage

### Generate Sample Data

To generate synthetic DNA/RNA data for testing:

```bash
python -m sft.data.generate_sample_data --output_dir ./data --num_sequences 1000 --num_classes 3 --seq_type dna
```

### Train a Classifier

To train a sequence classifier:

```bash
python -m sft.train_classifier \
    --data_path ./data/dna_classification_data.csv \
    --model_name evo2_7b \
    --num_classes 3 \
    --output_dir ./model_output \
    --num_epochs 5
```

### Make Predictions

To make predictions using a trained model:

```bash
python -m sft.predict \
    --data_path ./new_sequences.csv \
    --model_path ./model_output/checkpoint_epoch_5.pt \
    --model_name evo2_7b \
    --num_classes 3 \
    --output_path ./predictions.csv
```

### Run the Demo

For a complete demonstration of the workflow:

```bash
python -m sft.demo \
    --model_name evo2_1b_base \
    --num_classes 2 \
    --num_sequences 200 \
    --output_dir ./demo_output
```

## Tests

Run the unit tests to verify the implementation:

```bash
python -m unittest discover -s sft/tests
```

## Customization

The framework can be customized in various ways:

1. **Model Size**: Choose different Evo2 models based on your requirements (1B, 7B, 40B).
2. **Pooling Strategy**: Select from mean, max, or cls pooling for sequence embeddings.
3. **Freezing**: Decide whether to freeze the base model weights or fine-tune all parameters.
4. **Custom Layers**: Extract embeddings from different layers of the model.

## License

This code is released under the same license as the Evo2 model.
