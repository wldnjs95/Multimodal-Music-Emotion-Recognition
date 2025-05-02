# utils/data_loader.py

from torch.utils.data import DataLoader
from data.datasets import MultimodalDataset, get_half_dataset

# Function to create data loaders for training, validation, and testing
def create_data_loaders(train_df, val_df, test_df, audio_folder_path, lyric_folder_path, batch_size=16, use_half_data=False):
    # Create dataset instances
    train_dataset = MultimodalDataset(train_df, audio_folder_path, lyric_folder_path)
    val_dataset = MultimodalDataset(val_df, audio_folder_path, lyric_folder_path)
    test_dataset = MultimodalDataset(test_df, audio_folder_path, lyric_folder_path)

    # Optionally use half of the training dataset (useful for ablation experiments)
    if use_half_data:
        train_dataset = get_half_dataset(train_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
