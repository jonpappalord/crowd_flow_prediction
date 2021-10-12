# from main import raw_training
from src.AdjNet.trainModel import train_and_evaluate
from read_geodataframe import load_dataset
import mlflow
import os
from src.AdjNet.utils.config import Config

experiment_name = " ".join(["bike","NN"])\
    
if __name__ == '__main__':
    # Setting MLFlow
    mlflow.set_experiment(experiment_name = experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    sample_time = "60min"
    tile_size = 1000

    if not os.path.exists(Config().DATAPATH+"/BikeNYC/df_grouped_tile"+str(tile_size)+"freq"+sample_time+".csv"): 
        load_dataset(tile_size, sample_time)

    time_steps = 60/float(sample_time.split("min")[0])
    nb_epoch = 150  # number of epoch at training stage

    batch_size = 16
    lr = 1e-4
    lr_decay = 0.96
    opt = "RMSprop"

    past_time = 11
    

    for tile_size in [750]:
        for sample_time in ["30min", "45min", "60min"]:
            time_steps = 60/float(sample_time.split("min")[0])
            train_and_evaluate(tile_size, sample_time, nb_epoch, exp, time_steps, batch_size, lr, lr_decay, opt, past_time)
