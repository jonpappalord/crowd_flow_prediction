
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
        
    # def adj_shape(self):
    #     return self.A.shape

    # def get_adj_matrix(self):
    #     return self.A

    def get_data(self):
        return self.data

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :, :self.n_his],  self.data[idx, :, self.n_his:]



# class Dataset(object):
#     def __init__(self, data, stats):
#         self.__data = data
#         self.mean = stats['mean']
#         self.std = stats['std']

#     def get_data(self, type):
#         return self.__data[type]

#     def get_stats(self):
#         return {'mean': self.mean, 'std': self.std}

#     def get_len(self, type):
#         return len(self.__data[type])

#     def z_inverse(self, type):
#         return self.__data[type] * self.std + self.mean

