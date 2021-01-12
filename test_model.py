
import mlflow

from models.deepst.BikeNYC.trainModel import train_and_evaluate

experiment_name = " ".join(["bike","NN"])\

if __name__ == '__main__':
    # Setting MLFlow
    mlflow.set_experiment(experiment_name = experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    sample_time = "60min"
    tile_size = 1500
    nb_epoch = 500  # number of epoch at training stage
    
    train_and_evaluate(tile_size, sample_time, nb_epoch, exp)