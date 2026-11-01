import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_sequences(X, y, seq_len=20):
    X_seq = []
    y_seq = []

    for i in range(seq_len - 1, len(X)):
        X_seq.append(X.iloc[i-seq_len+1:i+1].values)
        y_seq.append(y.iloc[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    )


def create_loader(X, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )