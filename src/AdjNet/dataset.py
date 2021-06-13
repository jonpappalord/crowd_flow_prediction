
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, n_his):
        super().__init__()
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.n_his = n_his

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

        return self.data[idx, :, :self.n_his],  self.data[idx, :, self.n_his:]
