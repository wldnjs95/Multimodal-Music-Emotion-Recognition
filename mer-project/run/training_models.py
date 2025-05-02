# scripts/ast_bert_training.py

import torch
from data.loaders.merge import load_merge_dataset
from scripts.train import train, get_model
from scripts.evaluate import evaluate, load_best_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define base path (update accordingly)
base_path = '/content/drive/MyDrive/colab/MER/MERGE_bi_balanced'

# Load MERGE dataset using dedicated loader
train_loader, val_loader, test_loader = load_merge_dataset(base_path)

# Set model type (change this for different models)
model_type = "ast_bert"  # options: ast_bert, cross_attention, gated_fusion, modality_attention

# Initialize model
model = get_model(model_type).to(device)

# Train model and save best model based on validation loss
train(model, train_loader, val_loader, epochs=10, model_type=model_type, lr=2e-5)

# Load the best saved model and evaluate on test data
load_best_model(model, model_type)
evaluate(model, test_loader)
