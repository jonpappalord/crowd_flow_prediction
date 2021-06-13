import os
import zipfile
import numpy as np
import torch

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


# def generate_dataset(X, num_timesteps_input, num_timesteps_output):
#     """
#     Takes node features for the graph and divides them into multiple samples
#     along the time-axis by sliding a window of size (num_timesteps_input+
#     num_timesteps_output) across it in steps of 1.
#     :param X: Node features of shape (num_vertices, num_features,
#     num_timesteps)
#     :return:
#         - Node features divided into multiple samples. Shape is
#           (num_samples, num_vertices, num_features, num_timesteps_input).
#         - Node targets for the samples. Shape is
#           (num_samples, num_vertices, num_features, num_timesteps_output).
#     """
#     # Generate the beginning index and the ending index of a sample, which
#     # contains (num_points_for_training + num_points_for_predicting) points
#     indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
#                in range(X.shape[2] - (
#                 num_timesteps_input + num_timesteps_output) + 1)]

#     # Save samples
#     features, target = [], []
#     for i, j in indices:
#         features.append(
#             X[:, :, i: i + num_timesteps_input].transpose(
#                 (0, 2, 1)))
#         target.append(X[:, 0, i + num_timesteps_input: j])

#     return torch.from_numpy(np.array(features)), \
#            torch.from_numpy(np.array(target))

def get_origin_destination_matrix(df):
    df['tile_ID_origin'] -= df['tile_ID_origin'].min()
    df['tile_ID_destination'] -= df['tile_ID_destination'].min()

    time = set()
    x_axis = int(df['tile_ID_origin'].max())+1
    y_axis = int(df['tile_ID_destination'].max())+1
    origin_dest = np.zeros([len(df['starttime'].unique()), x_axis, y_axis])
    # print("Shape origin Destination matrix: ",origin_dest.shape)
    t = -1

    for el in df.itertuples():
        if(el.starttime not in time):
            time.add(el.starttime)
            t += 1
        origin_dest[t, el.tile_ID_origin, el.tile_ID_destination] += el.flow
    
    return origin_dest