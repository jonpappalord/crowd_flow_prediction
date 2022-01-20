import numpy as np
import ast
import geopandas as gpd
import pandas as pd
import shapely.wkt
from operator import itemgetter
import skmob
from skmob.tessellation import tilers

from src.AdjNet.dataset import Dataset
from src.AdjNet.utils.config import Config

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def load_dataset(tile_size, sample_time):
    df = pd.read_csv(Config().DATAPATH+"/BikeNYC/df_grouped_tile"+str(tile_size)+"freq"+sample_time+".csv") # TODO Change datapath
    min_tile_id = df['tile_ID_origin'].min()

    df['tile_ID_origin'] -= df['tile_ID_origin'].min()
    df['tile_ID_destination'] -= df['tile_ID_destination'].min()

    time = set()
    time_orgin_dest = []
    x_axis = int(df['tile_ID_origin'].max())+1
    y_axis = int(df['tile_ID_destination'].max())+1
    origin_dest = np.zeros([len(df['starttime'].unique()), x_axis, y_axis])
    print("Shape origin Destination matrix: ", origin_dest.shape)
    t = -1

    for el in df.itertuples():
        if(el.starttime not in time):
            time.add(el.starttime)
            t += 1
        origin_dest[t, el.tile_ID_origin, el.tile_ID_destination] = el.flow

    # Removing self loops, i.e. putting 0s in the diagonal of the OD matrix
    for i in range(origin_dest.shape[0]):
        np.fill_diagonal(origin_dest[i,:,:],0)

    # Removing nodes without flows
    od_sum = np.add.reduce(origin_dest)
    od_matrix = origin_dest[:,~(od_sum==0).all(1),:]
    od_matrix = od_matrix[:,:,~(od_sum.T==0).all(1)]
    
    empty_indices = [i for i, x in enumerate((od_sum==0).all(1)) if x]

    return od_matrix, empty_indices, min_tile_id

def split_and_scale(X, time_steps, n_his, n_pred):
    day_slot = int(24*time_steps)

    n_frame = n_his+n_pred
    n_route = X.shape[1]
    C_0 = X.shape[2]

    print("Len of day slot is: ", day_slot)
    days_test = 10
    
    data_dev = X[:-days_test*day_slot]
    print("Developer set shape: ", data_dev.shape[0])
    days_train = int(data_dev.shape[0]/day_slot*0.8)
    len_train = days_train*day_slot
    days_val = int(data_dev.shape[0]/day_slot*0.2)
    len_val = days_val*day_slot
    data_train = data_dev[:len_train]
    data_val = data_dev[len_train:]
    data_test = X[-days_test*day_slot:]

    print("Dataset is divided in training: ", data_train.shape, ", validation: ", data_val.shape, ", test: ", data_test.shape)

    seq_train = seq_gen(days_train, data_train, 0, n_frame, n_route, day_slot, C_0)
    seq_val = seq_gen(days_val, data_val, 0, n_frame, n_route, day_slot, C_0)
    seq_test = seq_gen(days_test, data_test, 0, n_frame, n_route, day_slot, C_0)
    
    print("Data test: ", data_test[n_frame-1:].shape, "Seq test: ", seq_test.shape)
  
    x_train = seq_train
    x_val = seq_val
    x_test = seq_test

    print("Shape Test: ", x_test.shape)

    train_dataset = Dataset(np.transpose(x_train.reshape(seq_train.shape), (0, 2, 1, 3)), n_his)
    val_dataset = Dataset(np.transpose(x_val.reshape(seq_val.shape), (0, 2, 1, 3)), n_his)
    test_dataset = Dataset(np.transpose(x_test.reshape(seq_test.shape), (0, 2, 1, 3)), n_his)
    
    return train_dataset, val_dataset, test_dataset


def get_matrix_mapping(tile_size):
    # tessellation = tilers.tiler.get("squared", base_shape="Manhattan, New York City, USA", meters=tile_size)
    tessellation = pd.read_csv(Config().DATAPATH+"/Tessellation_"+str(tile_size)+"m.csv")
    tessellation['geometry'] = [shapely.wkt.loads(el) for el in tessellation.geometry]
    tessellation = gpd.GeoDataFrame(tessellation, geometry='geometry')

    # list_positions = [np.array(el) for el in tessellation['position']]
    # list_positions = np.array(sorted(list_positions,key=itemgetter(1)))
    list_positions = [np.array(ast.literal_eval(el)) for el in tessellation['position']]
    list_positions = np.array(sorted(np.array(list_positions),key=itemgetter(1)))
        
    max_x = list_positions[:, 0].max()
    max_y = list_positions[:, 1].max()

    pos_set = set()
    new_value = max_y +1
    for i, pos in enumerate(list_positions[:, 1]):
        if pos not in pos_set:
            new_value -= 1
            pos_set.add(pos)
        list_positions[i, 1] = new_value
        
    tessellation['positions'] = list(sorted(list_positions, key=itemgetter(0)))
    # skmob.utils.plot.plot_gdf(tessellation, popup_features=['tile_ID', 'positions'])


    matrix_mapping = {el[0]:el[1] for el in zip(tessellation['tile_ID'], tessellation['positions'])}
    y_max = np.array(list(tessellation['positions']))[:,1].max()+1
    x_max = np.array(list(tessellation['positions']))[:,0].max()+1

    return matrix_mapping, x_max, y_max


def restore_od_matrix_pred(OD_matrix, empty_indices):
        for idx in empty_indices:
            OD_matrix = np.insert(OD_matrix, idx, np.zeros([OD_matrix.shape[0], 1, OD_matrix.shape[1]]), 1)
            OD_matrix = np.insert(OD_matrix, idx, np.zeros([OD_matrix.shape[0], OD_matrix.shape[1], 1]), 3)
        return OD_matrix


def OD_matrix_to_map(OD_matrix, mapping, offset, map_shape):
    map_matrix = np.zeros(map_shape)
    for i in range(OD_matrix.shape[1]): # origin
        for j in range(OD_matrix.shape[3]): # destination
            x, y = mapping[(j+offset)]
            map_matrix[:, x, y, 0, :] += OD_matrix[:, i, :, j]#.numpy() # Inflow
            x, y = mapping[(i+offset)]
            map_matrix[:, x, y, 1, :] += OD_matrix[:, i, :, j]#.numpy() # Outflow
    return map_matrix


def remove_empty_rows(X_dataset, flows):
    X_new = []
    X_sum = []
    for i in range(flows):
        X_new.append(X_dataset[:,:,:,i])
        X_sum.append(np.add.reduce(X_new[i]))

        X_new[i] = X_new[i][:,~(X_sum[i]==0).all(1)]    # Removing empty rows
        X_new[i] = X_new[i][:,:,~(X_sum[i].T==0).all(1)]    # Removing empty columns

    X_dataset = np.empty([X_dataset.shape[0], X_new[0].shape[1], X_new[0].shape[2], flows])

    for i in range(flows):
        X_dataset[:,:,:,i] = X_new[i]

    return X_dataset, (~(X_sum[i]==0).all(1), ~(X_sum[i].T==0).all(1))


def to_2D_map(actual, predicted, matrix_mapping, empty_indices, min_tile_id, x_max, y_max):
    # actual = restore_od_matrix_pred(actual, empty_indices)
    # predicted = restore_od_matrix_pred(predicted, empty_indices)

    actual_map = OD_matrix_to_map(actual, matrix_mapping, min_tile_id, [actual.shape[0], x_max, y_max, 2, 1])
    predicted_map = OD_matrix_to_map(predicted, matrix_mapping, min_tile_id, [predicted.shape[0], x_max, y_max, 2, 1])

    # Removing rows and columns with no flows
    actual_map, non_empty_shape = remove_empty_rows(actual_map[:,:,:,:,0], 2)
    predicted_map = predicted_map[:, non_empty_shape[0], :, :]
    predicted_map = predicted_map[:, :, non_empty_shape[1], :, :]
    predicted_map = predicted_map[:, :, :, :, 0]

    return actual_map, predicted_map