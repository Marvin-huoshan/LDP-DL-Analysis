"""Dataset classes and data loading utilities."""

from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AttackerDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           StandardScaler, np.ndarray]:
    X = df[feature_cols].values
    y = df['label'].values
    
    # Get indices for later sensitivity analysis
    indices = np.arange(len(df))
    
    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler, test_idx


def prepare_data_by_dataset_type(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_type: str,
    test_type: str,
    eval_type: str = None,
    random_state: int = 42
) -> dict:
    # Filter by dataset_type
    train_mask = df['dataset_type'] == train_type
    test_mask = df['dataset_type'] == test_type

    if not train_mask.any():
        raise ValueError(f"No rows found for train dataset_type='{train_type}'")
    if not test_mask.any():
        raise ValueError(f"No rows found for test dataset_type='{test_type}'")
    
    print(f"Found {df[train_mask].shape[0]} rows for train dataset_type='{train_type}'")
    print(f"Found {df[test_mask].shape[0]} rows for test dataset_type='{test_type}'")

    train_df = df[train_mask].sample(frac=1, random_state=random_state).reset_index()
    test_df = df[test_mask].sample(frac=1, random_state=random_state).reset_index()

    X_train_raw = train_df[feature_cols].values
    y_train = train_df['label'].values
    train_indices = train_df['index'].to_numpy()

    X_test_raw = test_df[feature_cols].values
    y_test = test_df['label'].values
    test_indices = test_df['index'].to_numpy()

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Train ({train_type}): {len(X_train)} samples")
    print(f"Test  ({test_type}): {len(X_test)} samples")

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'train_indices': train_indices,
        'X_test': X_test,
        'y_test': y_test,
        'test_indices': test_indices,
        'scaler': scaler,
    }

    # Optional evaluation set for three-way mode
    if eval_type is not None:
        eval_mask = df['dataset_type'] == eval_type
        if not eval_mask.any():
            raise ValueError(f"No rows found for eval dataset_type='{eval_type}'")

        eval_df = df[eval_mask].sample(frac=1, random_state=random_state).reset_index()
        X_eval_raw = eval_df[feature_cols].values
        y_eval = eval_df['label'].values
        eval_indices = eval_df['index'].to_numpy()

        X_eval = scaler.transform(X_eval_raw)

        print(f"Eval  ({eval_type}): {len(X_eval)} samples")

        result['X_eval'] = X_eval
        result['y_eval'] = y_eval
        result['eval_indices'] = eval_indices

    return result
