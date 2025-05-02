# data/loaders/merge.py

import os
import pandas as pd
from utils.data_loader import create_data_loaders

def load_merge_dataset(base_path, batch_size=16, use_half_data=False):
    info_folder = os.path.join(base_path, 'merged')
    data_path = os.path.join(base_path, 'bi_balanced_data')

    test_df = pd.read_csv(os.path.join(info_folder, 'merged_balanced_test_15.csv'))
    train_df = pd.read_csv(os.path.join(info_folder, 'merged_balanced_train_70.csv'))
    val_df = pd.read_csv(os.path.join(info_folder, 'merged_balanced_val_15.csv'))

    audio_path = os.path.join(data_path, 'audio')
    lyric_path = os.path.join(data_path, 'lyrics')

    return create_data_loaders(train_df, val_df, test_df, audio_path, lyric_path, batch_size, use_half_data)
