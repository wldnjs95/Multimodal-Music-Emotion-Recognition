# data/loaders/pmemo.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_loader import create_data_loaders

def extract_lyrics_from_lrc(lrc_path):
    with open(lrc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lyrics = []
    for line in lines:
        if ']' in line:
            parts = line.strip().split(']')
            if len(parts) > 1:
                text = parts[1].strip()
                if text:
                    lyrics.append(text)
    return " ".join(lyrics)


def load_pmemo_dataset(base_path, batch_size=16):
    metadata_path = os.path.join(base_path, 'metadata.csv')
    lyrics_dir = os.path.join(base_path, 'lyrics')
    audio_dir = os.path.join(base_path, 'chorus')

    # Extract lyrics
    lyrics_data = []
    for file in os.listdir(lyrics_dir):
        if file.endswith(".lrc"):
            file_path = os.path.join(lyrics_dir, file)
            text = extract_lyrics_from_lrc(file_path)
            if len(text) > 0:
                lyrics_data.append({
                    "fileName": file,
                    "lyrics": text
                })

    df_lyrics = pd.DataFrame(lyrics_data)
    df_lyrics["fileName"] = df_lyrics["fileName"].str.replace(".lrc", ".mp3")

    # Load metadata and emotion
    df_meta = pd.read_csv(os.path.join(base_path, 'metadata.csv'))
    df_emotion = pd.read_csv(os.path.join(base_path, 'annotations/static_annotations.csv'))
    df_emotion.rename(columns={
        "Arousal(mean)": "arousal",
        "Valence(mean)": "valence"
    }, inplace=True)

    # Merge all
    df = df_meta.merge(df_lyrics, on="fileName", how="inner")
    df = df.merge(df_emotion[["musicId", "valence", "arousal"]], on="musicId", how="left")

    df["metadata_text"] = df.apply(
        lambda row: f"{row['title']} by {row['artist']} from the album {row['album']}", axis=1)

    # Filter files that exist
    def file_exists(row):
        file_id = row['fileName'].replace(".mp3", "")
        audio_path = os.path.join(audio_dir, f"{file_id}.mp3")
        lyric_path = os.path.join(lyrics_dir, f"{file_id}.lrc")
        return os.path.exists(audio_path) and os.path.exists(lyric_path)

    df = df[df.apply(file_exists, axis=1)].reset_index(drop=True)
    df = df.dropna(subset=["valence", "arousal"]).reset_index(drop=True)

    # Split data
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    return create_data_loaders(df_train, df_val, df_test, audio_dir, lyrics_dir, batch_size)