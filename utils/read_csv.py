import igraph
import fnmatch
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pickle
import skmob
from shapely.geometry import Point
from skmob.tessellation import tilers
import time
from zipfile import ZipFile

# FILENAME = "utils/2013-12 - Citi Bike trip data.csv"

def getIndexPosition(tessellation, position):
    for i, row in enumerate(tessellation):
        if row.contains(position):
            return i

def read_csv(
        tessellation = None, 
        list_feature = ["starttime", "start station latitude", "start station longitude", "end station latitude", "end station longitude"],
        sample_time = "60min",
        dataset_file = "data/BikeNYC/BikeNYC.zip",
        CACHE = True,
        # filename_df = "NYC1500m60min.pkl",
        filename_df = "NYC_temp.pkl"
        ):
    """
    Given a csv file and a tessellation, returns a dataframe contaning origin destination and the time

    Parameters
    ----------
    - tessellation = None, 
    - list_feature = ["starttime", "start station latitude", "start station longitude", "end station latitude", "end station longitude"],
    - sample_time = "60min",
    - dataset_file = "data/BikeNYC/BikeNYC.zip",
    - CACHE = True,
    - filename_df = "NYC_caso.pkl": A list containing the neural network inputs 

    Returns
    -------
    - df: the csv file as a Pandas DataFrame
    """

    if os.path.exists(filename_df) and CACHE:
        print("Loading the existing dataframe: ", filename_df)
        with open(filename_df, "rb") as input_file:
            df = pickle.load(input_file)
    else:
        if dataset_file.endswith('.zip'):
            with ZipFile(dataset_file) as zipfiles:
                file_list = zipfiles.namelist()
                
                #get only the csv files
                csv_files = fnmatch.filter(file_list, "*.csv")
                
                #iterate with a list comprehension to get the individual dataframes
                data = [pd.read_csv(zipfiles.open(file_name)) for file_name in csv_files]
                df = pd.concat(data)
        else:
            df = pd.read_csv(dataset_file, sep=',')

        print("DataFrame read!")
        df = df[list_feature]
        df['origin'] = df.apply(lambda row: Point(row['start station longitude'], row['start station latitude']), axis=1)
        df['destination'] = df.apply(lambda row: Point(row['end station longitude'], row['end station latitude']), axis=1)

        poligons = tessellation['geometry']

        df['origin'] = df.apply(lambda row: getIndexPosition(poligons, row['origin']), axis=1)
        print("Origin column added") 
        df['destination'] = df.apply(lambda row: getIndexPosition(poligons, row['destination']), axis=1)
        print("Destination column added") 

        with open(filename_df, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    return df

def get_xy_location(tessellation):
    """
    Given a tessellation with longitude and latitude coordinates, 
    returns a list of positions in a squared matrix for each square and its size
    """
    centroids = [el.centroid for el in tessellation['geometry']]

    x_list, y_list = [], []
    for el in tessellation['geometry']:
        x_list.append(el.centroid.x)
        y_list.append(el.centroid.y)
    x_list = sorted(list(set(x_list)))
    y_list = sorted(list(set(y_list)))

    x_dict = {coordinate:val for val, coordinate in enumerate(x_list)}
    y_dict = {coordinate:val for val, coordinate in enumerate(y_list)}

    return [(x_dict[centroid.x], y_dict[centroid.y]) for centroid in centroids], len(x_list), len(y_list)

def get_xy_map(df, m_shape):
    """
    Given a dataframe and a matrix shape
    Returns a matrix with given sizes and the timestamp for each record of the dataframe, corresponding to the given DataFrame
    """
    X = np.zeros(m_shape)
    time_samples = set()
    t = -1
    for metadata, flow in df.iterrows():
        if metadata[0] not in time_samples:
            time_samples.add(metadata[0])
            t += 1
        x = metadata[1][0]
        y = metadata[1][1]
        X[t][int(x)][int(y)] = flow['flow']
    return X, time_samples

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

    return X_dataset


def df_to_matrix(df, tessellation, list_features=["origin", "destination", "starttime"], sample_time="60min", flows=2):
    """"
    Given a dataframe and a tessellation returns a temporal matrix and its timeseries
    """
    df.dropna(inplace=True)
    fdf = df[list_features]
    fdf['origin'] = fdf['origin'].astype(int)
    fdf['destination'] = fdf['destination'].astype(int)
    fdf['flow'] = 1

    fdf = skmob.FlowDataFrame(fdf,
                            tessellation=tessellation,
                            origin="origin",
                            destination="destination",
                            tile_id='tile_ID')

    flow_df = fdf[['starttime','origin','destination', 'flow']]
    flow_df['starttime'] = pd.to_datetime(flow_df.starttime)
    flow_df['origin'] = flow_df['origin'].astype(int)
    flow_df['destination'] = flow_df['destination'].astype(int)

    print("Getting xy location")
    tessellation['map_point'], x_size, y_size = get_xy_location(tessellation)

    # Create a dict to map each tile_ID to a point of the xy_map
    tile_to_xy = {key:val for (key,val) in zip(tessellation['tile_ID'], tessellation['map_point'])}

    # Transform the origin and the destination from tile_IDs to position of the xy_map
    flow_df['origin'] = [tile_to_xy[str(el)] for el in flow_df['origin']]
    flow_df['destination'] = [tile_to_xy[str(el)] for el in flow_df['destination']]

    n_timestamps = flow_df.groupby(pd.Grouper(key='starttime', freq=sample_time)).ngroups

    f_out = flow_df.groupby([pd.Grouper(key='starttime', freq=sample_time),'origin']).sum()
    f_in = flow_df.groupby([pd.Grouper(key='starttime', freq=sample_time),'destination']).sum()

    # Filling numpy array 
    X_dataset = np.empty([n_timestamps, x_size, y_size, flows])
    
    print("Getting xy map")
    for i in range(flows):
        X_dataset[:,:,:,i], time_samples = get_xy_map(f_in, [n_timestamps, x_size, y_size])
    # TODO: delete:
    # X_dataset[:,:,:,0], _ = get_xy_map(f_in, [n_timestamps, x_size, y_size]) # Inflow
    # X_dataset[:,:,:,1], time_samples = get_xy_map(f_out, [n_timestamps, x_size, y_size]) # Outflow

    # Reduce size, by removing rows and columns without data
    X_dataset = remove_empty_rows(X_dataset, flows)

    time_samples = list(time_samples)

    # Adapting the time_samples list
    time_string = [str(int(str(time_sample).replace("-", "").replace(" ","").replace(":", "")[:10])+1).encode('utf-8') for time_sample in time_samples]

    return X_dataset, time_string

def csv_stub(
            tile_size = 1500,
            sample_time = "60min",
            list_feature = ["starttime", "start station latitude", "start station longitude", "end station latitude", "end station longitude"],
            dataset_file = "data/BikeNYC/BikeNYC.zip",
            CACHE = True
        ):
    """
    Returns a temporal matrix and its timeseries
    """
    print("Reading csv data")
    ts = time.time()
    filename_df = "NYC"+str(tile_size)+sample_time+"_df.pkl"
    tessellation = tilers.tiler.get("squared", base_shape="Manhattan, New York City, USA", meters=tile_size)
    df = read_csv(tessellation, dataset_file = dataset_file, sample_time=sample_time, filename_df=filename_df)
    X_data, time_stamp = df_to_matrix(df, tessellation, sample_time=sample_time)

    print("\nelapsed time (read csv data): %.3f seconds\n" % (time.time() - ts))
    return X_data, time_stamp
    

"""
tessellation = tilers.tiler.get("squared", base_shape="Manhattan, New York City, USA", meters=1500)
print("Reading csv")
df = read_csv(tessellation)
print("Read!")
print("Parsing")
X_dataset, time_string = df_to_matrix(df, tessellation)
"""

"""
######

T = 24
PeriodInterval = 1
TrendInterval = 7
len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
c_list = range(1, len_closeness+1)
p_list = [PeriodInterval * T * j for j in range(1, len_period+1)]
t_list = [TrendInterval * T * j for j in range(1, len_trend+1)]

# from utils.load_datasets.STMatrix import STMatrix

# st_matrix = STMatrix()



from utils.config import Config
import h5py



# parameter
DATAPATH = Config().DATAPATH
def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


#  load data

from utils.load_datasets import BikeNYC
nb_flow = 2
len_test = 10

import pickle
import numpy as np

# from . import load_stdata
from utils.minmax_normalization import MinMaxNormalization
from utils.preprocessing_functions import remove_incomplete_days, timestamp2vec
from utils.config import Config
from utils.load_datasets.STMatrix import STMatrix
np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH


def load_data(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None, preprocess_name='preprocessing.pkl', meta_data=True, stdata_file=None):
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    if stdata_file is None:
        data, timestamps = load_stdata(os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5'))
        # print(timestamps)
        # remove a certain day which does not have 24 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :, :, :nb_flow]
        data[data < 0] = 0.

        # Reorder data to have CWH
        data = np.transpose(data, (0, 2, 3, 1))
    else:
        data, timestamps = stdata_file

    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        print(timestamps[0])
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    # load meta feature
    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            preprocess_name='my_preprocessing.pkl', meta_data=True, 
            # stdata_file=(X_dataset, [str(time_sample).replace("-", "").replace(" ","").replace(":", "")[:10] for time_sample in time_samples]))
            stdata_file= (X_dataset, time_string))

"""