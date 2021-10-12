import ast
import fnmatch
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from operator import itemgetter
import pandas as pd
import requests
import skmob
from shapely.geometry import Point
import shapely.wkt
from skmob.tessellation import tilers
import time
from src.AdjNet.utils.config import Config
from zipfile import ZipFile

# dataset_file = "data/BikeNYC/BikeNYC.zip"
dataset_directory = "data/BikeNYC/"


tile_size = 1000
sample_time = '60min'

def load_dataset(tile_size=1000, sample_time='60min'):
    for month in [4, 5, 6, 7, 8, 9]:
        if not os.path.isfile(dataset_directory+"20140"+str(month)+"-citibike-tripdata.zip"):
            url = 'https://s3.amazonaws.com/tripdata/20140'+str(month)+'-citibike-tripdata.zip'
            r = requests.get(url, allow_redirects=True)
            open(dataset_directory+"20140"+str(month)+"-citibike-tripdata.zip", 'wb').write(r.content)
            print("Downloaded month: ", month)

    print("Loading data...")
    zip_files = [f for f in os.listdir(dataset_directory) if f.endswith('.zip')]
    data = [pd.read_csv(dataset_directory+file_name) for file_name in zip_files]
    df = pd.concat(data)
    # if dataset_file.endswith('.zip'):
    #     with ZipFile(dataset_file) as zipfiles:
    #         file_list = zipfiles.namelist()
            
    #         #get only the csv files
    #         csv_files = fnmatch.filter(file_list, "*.csv")
            
    #         #iterate with a list comprehension to get the individual dataframes
    #         data = [pd.read_csv(zipfiles.open(file_name)) for file_name in csv_files]
    #         df = pd.concat(data)
    # else:
    #     df = pd.read_csv(dataset_file, sep=',')
    print("Data loaded...")
    print("Preprocessing")


    # tessellation = tilers.tiler.get("squared", base_shape="Manhattan, New York City, USA", meters=tile_size)
    tessellation = pd.read_csv(Config().DATAPATH+"/Tessellation_"+str(tile_size)+"m.csv")
    tessellation['geometry'] = [shapely.wkt.loads(el) for el in tessellation.geometry]
    tessellation = gpd.GeoDataFrame(tessellation, geometry='geometry')

    print("Tessellato")

    # tessellation['position'] contains che position in a matrix
    # The origin is located on the bottom left corner
    # We need to locate it on the top left corner

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
    gdf_grouped.to_csv(Config().DATAPATH+"/BikeNYC/df_grouped_tile"+str(tile_size)+"freq"+sample_time+".csv")