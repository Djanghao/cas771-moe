# Training Runs Directory

This directory contains the results of different training runs for the Mixture of Experts (MoE) model. Each subdirectory represents a separate training run with a timestamp to prevent overwriting previous results.

## Directory Structure

```
<model_name>_<timestamp>/
├── config.json             # Configuration parameters used for training
├── best_model.pth          # Best model weights during training
├── training_log.txt        # Detailed training logs (loss, accuracy, etc.)
├── learning_curve.png      # Learning curve updated in real-time during training
├── final_learning_curve.png # Final learning curve after training completion
└── expert_analysis_*.png   # Visualizations of expert specialization at various epochs
```

*Note: For detailed information about parameters, loss functions, and training process, please refer to the main README.md file in the project root directory.*
