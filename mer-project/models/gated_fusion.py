# models/gated_fusion.py

import torch
from torch import nn
from transformers import ASTModel, BertModel

class ASTBERTGatedFusionRegressor(nn.Module):
    def __init__(self):
        super(ASTBERTGatedFusionRegressor, self).__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gate_fc = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Sigmoid()
        )
        self.regressor = nn.Linear(768, 2)

    def forward(self, audio_inputs, text_inputs):
        # Extract [CLS] token from audio features using AST
        audio_cls = self.ast(**audio_inputs).last_hidden_state[:, 0, :]
        # Extract [CLS] token from text features using BERT
        text_cls = self.bert(**text_inputs).last_hidden_state[:, 0, :]
        # Generate gating values to weigh audio and text features dynamically
        gate_values = self.gate_fc(torch.cat([audio_cls, text_cls], dim=1))
        # Perform gated fusion of audio and text features
        fused_features = gate_values * audio_cls + (1 - gate_values) * text_cls
        # Regression to valence and arousal
        output = self.regressor(fused_features)
        return output
