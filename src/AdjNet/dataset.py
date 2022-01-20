
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, n_his, transform=None, target_transform=None):
        super().__init__()
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.n_his = n_his
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

    def shape(self):
        return self.data.shape
        
    def get_data(self):
        return self.data

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hist_data = self.data[idx, :, :self.n_his]
        label_data = self.data[idx, :, self.n_his:]

        if self.transform:
            hist_data = self.transform(hist_data)

        if self.target_transform:
            label_data = self.target_transform(label_data)

        return hist_data, label_data
