# Multimodal Music Emotion Recognition

A Deep Learning project that predicts the emotional content of music by integrating audio and lyrical information, using multimodal models based on Transformer architectures.

---

## Goal

Predict continuous emotional values of **Valence** (positivity) and **Arousal** (energy) from songs by leveraging both audio signals and lyrics.

## Dataset

- **MERGE Dataset** ([Zenodo Link](https://zenodo.org/record/13939205))
  - 2000 songs
  - Audio: 30-second MP3 clips
  - Lyrics: Plain text (.txt)
  - Valence and Arousal annotations by multiple raters
  - Train/Val/Test split: 70% / 15% / 15%

## Methodology

### Models & Approaches
- **Single-modal Baseline Models**:
  - AST-only
  - BERT-only

- **Multimodal Fusion Models**:
  - Concat (Baseline Model)
  - Cross-Attention Fusion (Best Performing Model)
  - Modality Attention
  - Gated Fusion

### Best Model: Cross-Attention Fusion
- Audio features extracted by **Audio Spectrogram Transformer (AST)**
- Lyrics features extracted by **BERT**
- Cross-attention mechanism to integrate multimodal information effectively

## Results

The Cross-Attention Fusion model outperformed other baselines, achieving the highest correlation and agreement scores for valence and arousal prediction.

*table image* 

## Key Insights

- Lyrics predominantly influence **Valence**.
- Audio features predominantly influence **Arousal**.
- Cross-attention effectively merges complementary emotional cues from audio and text.

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- AST, BERT

## Project Structure
```
MER-PROJECT
├── data
│   ├── loaders
│   └── datasets.py
├── models
│   ├── ast_bert.py
│   ├── cross_attention.py
│   ├── gated_fusion.py
│   └── modality_attention.py
├── run
│   └── training_models.py
├── scripts
│   ├── evaluate.py
│   └── train.py
└── utils
    └── data_loader.py
```

## Future Work
- Apply debiasing strategies to handle dataset biases towards older music
- Enhance metaphorical and nuanced language interpretation in lyrics

---

*Project by Team Deep Learner.*
