# scripts/evaluate.py

import torch
import glob
from torch import nn
from torch.utils.data import DataLoader
from data.datasets import MultimodalDataset
from models.ast_bert import ASTBERTRegressor
from models.cross_attention import ASTBERTCrossAttentionRegressor
from models.gated_fusion import ASTBERTGatedFusionRegressor
from models.modality_attention import ASTBERTModalityAttentionRegressor

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to get model by type
def get_model(model_type):
    if model_type == "ast_bert":
        return ASTBERTRegressor()
    elif model_type == "cross_attention":
        return ASTBERTCrossAttentionRegressor()
    elif model_type == "gated_fusion":
        return ASTBERTGatedFusionRegressor()
    elif model_type == "modality_attention":
        return ASTBERTModalityAttentionRegressor()
    else:
        raise ValueError("Invalid model type")

# Load the best model based on model type
def load_best_model(model, model_type):
    model_files = glob.glob(f"{model_type}_epoch_*_best_model.pt")
    if not model_files:
        raise FileNotFoundError(f"No saved model found for model type: {model_type}")
    # Load the latest saved model
    latest_model = sorted(model_files)[-1]
    model.load_state_dict(torch.load(latest_model))
    print(f"Loaded model: {latest_model}")

# Evaluation loop function
def evaluate(model, test_loader):
    model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for audio_inputs, text_inputs, labels in test_loader:
            audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, text_inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")