# models/cross_attention.py

import torch
from torch import nn
from transformers import ASTModel, BertModel

class ASTBERTCrossAttentionRegressor(nn.Module):
    def __init__(self):
        super(ASTBERTCrossAttentionRegressor, self).__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.regressor = nn.Linear(768, 2)

    def forward(self, audio_inputs, text_inputs):
        # Extract audio sequence features from AST
        audio_sequence = self.ast(**audio_inputs).last_hidden_state
        # Extract text sequence features from BERT
        text_sequence = self.bert(**text_inputs).last_hidden_state
        # Apply cross-attention: text features attend to audio features
        cross_attention_output, _ = self.cross_attention(query=text_sequence, key=audio_sequence, value=audio_sequence)
        # Use the first token ([CLS]) after cross-attention
        fused_features = cross_attention_output[:, 0, :]
        # Regression to valence and arousal
        output = self.regressor(fused_features)
        return output