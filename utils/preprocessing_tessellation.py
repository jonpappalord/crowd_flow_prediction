import geopandas as gpd
import igraph
import numpy as np
import pandas as pd
import pickle
import skmob
from skmob.tessellation import tilers
from shapely.geometry import Point

CACHE = True
filename_fdf = 'fdf.pkl'

def getIndexPosition(tessellation, position):
    for i, row in enumerate(tessellation):
        if row.contains(position):
            return i

def generate_fdf(FILENAME = "utils/2013-12 - Citi Bike trip data.csv"):
    
    if CACHE :
        with open(filename_fdf, "rb") as input_file:
            fdf = pickle.load(input_file)
    else: 
        url_tess = skmob.utils.constants.NY_COUNTIES_2011
        tessellation = tilers.tiler.get("squared", base_shape="New York City, New York", meters=1500)

        df = pd.read_csv(FILENAME, sep=',')
        df['origin'] = df.apply(lambda row: Point(row['start station longitude'], row['start station latitude']), axis=1)
        df['destination'] = df.apply(lambda row: Point(row['end station longitude'], row['end station latitude']), axis=1)

        poligons = tessellation['geometry']

        # Long time operation
        df['origin'] = df.apply(lambda row: getIndexPosition(poligons, row['origin']), axis=1)
        df['destination'] = df.apply(lambda row: getIndexPosition(poligons, row['destination']), axis=1)

        fdf = skmob.FlowDataFrame(df,
                                tessellation=tessellation,
                                origin="origin",
                                destination="destination",
                                tile_id='tile_ID')

        fdf['flow'] = 1
        with open('fdf.pkl', 'wb') as f:
            pickle.dump(fdf, f, protocol=pickle.HIGHEST_PROTOCOL)


    flow_df = fdf[['starttime','origin','destination', 'flow']]

    flow_df['starttime'] = pd.to_datetime(fdf.starttime)
    flow_df['origin'] = flow_df['origin'].astype(int)-int(flow_df['origin'].min())
    flow_df['destination'] = flow_df['destination'].astype(int)-int(flow_df['destination'].min())

    return flow_df

def generate_adj_matrix(f, max_origin, max_destination, min_origin=271, min_destination=271):
    f = f.groupby([pd.Grouper(key='starttime', freq='30min'),'origin','destination']).sum()
    time = set()
    time_orgin_dest = []
    origin_dest = np.zeros(
                [max_origin+1-min_origin, 
                max_destination+1-min_destination])
    flow = list(f['flow'])

    for i, el in enumerate(f.index):
        if(el[0] not in time):
            time.add(el[0])
            time_orgin_dest.append(origin_dest)
            origin_dest = np.zeros(
                [max_origin+1-min_origin, 
                max_destination+1-min_destination])
        origin_dest[el[1],el[2]] = flow[i]
    return time_orgin_dest


fdf = generate_fdf()
m = generate_adj_matrix(fdf, int(fdf['origin'].max()), int(fdf['destination'].max()), int(fdf['origin'].min()), int(fdf['destination'].min()))
g = igraph.Graph.Adjacency(list(m[8]))

print(g.clusters(mode="weak").sizes())
