# data/datasets.py

import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset, Subset
from transformers import ASTFeatureExtractor, BertTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, df, audio_dir, lyric_dir):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.lyric_dir = lyric_dir
        self.ast_proc = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        audio_id, lyric_id = row['Song'], row['Lyric_Song']
        quadrant, valence, arousal = row['Quadrant'], row['Valence'], row['Arousal']

        audio_path = os.path.join(self.audio_dir, quadrant, f"{audio_id}.mp3")
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        audio_inputs = self.ast_proc(waveform, sampling_rate=16000, return_tensors="pt")
        audio_inputs = {k: v.squeeze(0) for k, v in audio_inputs.items()}

        lyric_path = os.path.join(self.lyric_dir, quadrant, f"{lyric_id}.txt")
        with open(lyric_path, 'r') as f:
            text = f.read()
        text_inputs = self.bert_tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        label = torch.tensor([valence, arousal], dtype=torch.float)
        return audio_inputs, text_inputs, label


# Code for using only half of the training dataset (useful for ablation experiments)
def get_half_dataset(full_dataset, seed=42):
    half_len = len(full_dataset) // 2
    random.seed(seed)
    half_indices = random.sample(range(len(full_dataset)), half_len)
    return Subset(full_dataset, half_indices)
