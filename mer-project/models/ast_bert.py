# models/ast_bert.py

import torch
from torch import nn
from transformers import ASTModel, BertModel

class ASTBERTRegressor(nn.Module):
    def __init__(self):
        super(ASTBERTRegressor, self).__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.regressor = nn.Linear(768 + 768, 2)

    def forward(self, audio_inputs, text_inputs):
        # Obtain audio feature representation by averaging AST hidden states
        audio_features = self.ast(**audio_inputs).last_hidden_state.mean(dim=1)
        # Obtain text feature representation from BERT CLS token
        text_features = self.bert(**text_inputs).last_hidden_state[:, 0, :]
        # Concatenate audio and text features
        fused_features = torch.cat([audio_features, text_features], dim=1)
        # Regression to valence and arousal
        output = self.regressor(fused_features)
        return output
