# models/modality_attention.py

import torch
from torch import nn
from transformers import ASTModel, BertModel

class ASTBERTModalityAttentionRegressor(nn.Module):
    def __init__(self):
        super(ASTBERTModalityAttentionRegressor, self).__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Modality attention applied after concatenation of modalities
        self.modality_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Final regression layer
        self.regressor = nn.Linear(768, 2)

    def forward(self, audio_inputs, text_inputs):
        # Encode audio features by averaging AST hidden states
        audio_features = self.ast(**audio_inputs).last_hidden_state.mean(dim=1)  # [B, 768]

        # Encode text features using BERT CLS token
        text_features = self.bert(**text_inputs).last_hidden_state[:, 0, :]      # [B, 768]

        # Concatenate modalities into a sequence: shape [B, 2, 768]
        modality_sequence = torch.stack([audio_features, text_features], dim=1)

        # Apply self-attention across modality sequence
        attention_output, _ = self.modality_attention(modality_sequence, modality_sequence, modality_sequence)

        # Use attention output from text modality position as fused representation
        fused_features = attention_output[:, 1, :]  # [B, 768]

        # Predict valence and arousal
        output = self.regressor(fused_features)
        return output