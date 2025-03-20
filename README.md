# Mixture of Experts (MoE) Training Framework

This repository contains a training framework for Mixture of Experts (MoE) models designed for multi-task learning.

## Features

- Automatic experiment management with timestamped directories
- Regular updates to learning curves (every 10 epochs)
- Comprehensive expert analysis and visualization
- Command-line argument support for flexible training configuration
- Dynamic temperature scaling and balanced expert initialization
- Advanced loss functions with diversity and balance penalties

## Directory Structure

Each experiment run creates a timestamped directory under `./runs/` with the following structure:

```
./runs/YYYYMMDD_HHMMSS/
├── config.json          # Configuration parameters for the run
├── weights/             # Model checkpoints
├── results/             # Learning curves and other metrics
└── analysis/            # Expert analysis visualizations and reports
```

## Model Architecture

### Feature Extractor
- Convolutional neural network with Squeeze-and-Excitation blocks
- Residual connections and batch normalization
- Adaptive average pooling to 512-dimensional feature vectors

### Expert Networks
- Number of experts configurable (default: 4)
- Two-layer MLP with batch normalization and dropout
- Hidden dimension: 256
- Dropout rate: 0.3

### Gating Network
- Temperature-scaled softmax gating
- Dynamic temperature annealing (initial: 2.0, final: 0.5)
- Noise injection during training (σ = 0.1)
- Balanced initialization for uniform expert utilization

## Training Parameters

### Basic Parameters
- `--seed`: Random seed for reproducibility (default: 100)
- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay for regularization (default: 1e-4)
- `--num-experts`: Number of experts in the MoE model (default: 4)
- `--patience`: Number of epochs for early stopping (default: 50)
- `--model-name`: Base name for saved model files (default: 'moe_combined')

### Advanced Parameters

#### Loss Function Weights
- `diversity_weight`: Controls expert diversity (default: 0.1)
  - Higher values encourage experts to specialize in different tasks
  - Penalizes redundant expert behavior
- `balance_weight`: Controls load balancing (default: 0.5)
  - Higher values encourage uniform expert utilization
  - Prevents expert collapse (where only few experts are used)

#### Temperature Parameters
- Initial temperature: 2.0
  - Higher initial temperature promotes exploration
  - Softens gating decisions early in training
- Final temperature: 0.5
  - Lower final temperature sharpens expert selection
  - Encourages more decisive routing decisions
- Temperature is annealed over training using cosine schedule

## Usage

Run the training script with desired parameters:

```bash
python main.py --epochs 200 --batch-size 32 --lr 0.001 --num-experts 4
```

## Components

- `main.py`: Entry point with argument parsing
- `train.py`: Training and evaluation logic with loss balancing
- `model.py`: Model architecture with temperature scaling
- `data_loader.py`: Data loading utilities
- `utils.py`: Experiment management and utility functions

## Outputs

- Model weights: Saved in the weights directory with accuracy and epoch information
- Learning curves: Generated every 10 epochs showing:
  - Training loss (including diversity and balance components)
  - Training accuracy
  - Test accuracy
- Expert analysis: 
  - Per-expert accuracy for each class
  - Expert utilization statistics
  - Gate activation patterns
  - Class specialization analysis
