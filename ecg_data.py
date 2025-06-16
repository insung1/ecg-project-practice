# ecg_data.py

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from config import BASE_PATH, VALID_PATIENT_IDS

def load_ecg_data():
    X_list, y_list = [], []

    for pid in VALID_PATIENT_IDS:
        data_path = os.path.join(BASE_PATH, f"heartbeat_{pid}_filtered_data.npy")
        label_path = os.path.join(BASE_PATH, f"heartbeat_{pid}_labels.npy")

        if os.path.exists(data_path) and os.path.exists(label_path):
            X = np.load(data_path)
            y = np.load(label_path)
            X_list.append(X)
            y_list.append(y)
        else:
            print(f"❌ {pid}번 환자 파일 누락. 건너뜀.")

    if len(X_list) == 0:
        raise ValueError("❌ 데이터가 없습니다.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if X_all.shape[1] == 1:
        X_all = np.transpose(X_all, (0, 1, 2))
    else:
        X_all = np.transpose(X_all, (0, 2, 1))

    return X_all, y_all

def create_dataloaders(X_all, y_all, batch_size):
    # 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Tensor 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, val_dataset, test_dataset
