# scripts/train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.datasets import MultimodalDataset, get_half_dataset
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

# Training loop function with best model saving
def train(model, train_loader, val_loader, epochs, model_type, lr=2e-5):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for audio_inputs, text_inputs, labels in train_loader:
            audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio_inputs, text_inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audio_inputs, text_inputs, labels in val_loader:
                audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                labels = labels.to(device)

                outputs = model(audio_inputs, text_inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_type}_epoch_{epoch+1}_best_model.pt")
