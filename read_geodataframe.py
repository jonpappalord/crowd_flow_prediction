import fnmatch
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from operator import itemgetter
import pandas as pd
import skmob
from shapely.geometry import Point
from skmob.tessellation import tilers
import time
from utils.config import Config
from zipfile import ZipFile

dataset_file = "data/BikeNYC/BikeNYC.zip"

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

tile_size = 1000
sample_time = '120min'

tessellation = tilers.tiler.get("squared", base_shape="Manhattan, New York City, USA", meters=tile_size)

# tessellation['position'] contains che position in a matrix
# The origin is located on the bottom left corner
# We need to locate it on the top left corner
list_positions = [np.array(el) for el in tessellation['position']]
list_positions = np.array(sorted(list_positions,key=itemgetter(1)))
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
skmob.utils.plot.plot_gdf(tessellation, popup_features=['tile_ID', 'positions'])


## Filtering the dataset using the relevant features

df = df[['starttime', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude', 'stoptime', 'bikeid']]

df_in = df[['starttime', 'start station latitude', 'start station longitude', 'bikeid']]
df_out = df[['end station latitude', 'end station longitude', 'stoptime', 'bikeid']]

gdf_in = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start station longitude'], df['start station latitude']))
gdf_in_join = gpd.sjoin(gdf_in, tessellation)
gdf_in_join = gdf_in_join[['starttime',	'end station latitude', 'end station longitude', 'stoptime', 'bikeid', 'tile_ID', 'positions']]


gdf_final = gpd.GeoDataFrame(gdf_in_join, geometry=gpd.points_from_xy(gdf_in_join['end station longitude'], gdf_in_join['end station latitude']))
gdf_final_join = gpd.sjoin(gdf_final, tessellation)
gdf_final_join = gdf_final_join[['starttime', 'stoptime', 'bikeid', 'tile_ID_left', 'positions_left', 'tile_ID_right', 'positions_right']]

gdf_final_join = gdf_final_join.rename(columns={"tile_ID_left": "tile_ID_origin", "positions_left": "origin", "tile_ID_right": "tile_ID_destination", "positions_right": "destination"})
gdf_final_join['starttime'] = pd.to_datetime(gdf_final_join['starttime'])
gdf_final_join = gdf_final_join.sort_values(by='starttime')


gdf_final_join['flow'] = 1
gdf = gdf_final_join[['starttime', 'tile_ID_origin', 'tile_ID_destination', 'flow']]

gdf_grouped = gdf.groupby([pd.Grouper(key='starttime', freq=sample_time), 'tile_ID_origin','tile_ID_destination']).sum()

matrix_mapping = {el[0]:el[1] for el in zip(tessellation['tile_ID'], tessellation['positions'])}

# Saving geodataframe
gdf_grouped.to_csv(Config().DATAPATH+"/BikeNYC/df_grouped_tile"+str(tile_size)+".csv")


## Creating the matrix map to test the STResNet

def f(row, matrix_mapping, X_dataset, timestamps):
    (t, o, d), f = row.name, row.flow       # time, origin, destination, flow
    if t not in timestamps:
        timestamps.add(t)
    time_idx = len(timestamps)-1
    # print(time_idx, f)
    x_out, y_out = matrix_mapping[str(o)]
    x_in, y_in = matrix_mapping[str(d)]

    X_dataset[time_idx][x_in][y_in][0] += f     # Inflow
    X_dataset[time_idx][x_out][y_out][1] += f   # Outflow

flows = 2
n_timestamps = gdf.groupby([pd.Grouper(key='starttime', freq=sample_time)]).ngroups # It can be done also by counting the days*24 
matrix_shape = (n_timestamps, max_x+1, max_y+1, flows)
X_dataset = np.zeros(matrix_shape)
timestamps = set()

gdf_grouped.apply(lambda row: f(row, matrix_mapping, X_dataset, timestamps), axis=1)

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

X = remove_empty_rows(X_dataset, 2)

gdf.groupby([pd.Grouper(key='starttime', freq=sample_time)]).sum()


from utils.config import Config
from utils.load_datasets import save_stdata
DATAPATH = Config().DATAPATH

filename_df = "MANHATTAN_SIZE"+str(tile_size)+"_TIME_"+sample_time+"_df.h5"


# time_string = [str(int(str(time_sample).replace("-", "").replace(" ","").replace(":", "")[:10])+1).encode('utf-8') for time_sample in timestamps]
time_string = [str(int(str(time_sample).replace("-", "").replace(" ","").replace(":", "")[:8])).encode('utf-8') for time_sample in timestamps]

time_string.sort()

date_set = set(time_string[0])
hour = 1
for i, date in enumerate(time_string):
    if date not in date_set:
        hour = 1
        date_set.add(date)
    time_string[i] += str(hour).encode('utf-8') if hour > 9 else (str(0)+str(hour)).encode('utf-8')
    hour += 1
    
save_stdata(os.path.join(DATAPATH, 'BikeNYC',filename_df), X, time_string)
print("Saved!")