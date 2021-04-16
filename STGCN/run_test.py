from main import raw_training
import mlflow

experiment_name = " ".join(["bike","NN"])\
    
if __name__ == '__main__':
    # Setting MLFlow
    mlflow.set_experiment(experiment_name = experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    sample_time = "60min"
    time_steps = 60/float(sample_time.split("min")[0])
    tile_size = 1000
    nb_epoch = 150  # number of epoch at training stage

    batch_size = 16
    lr = 1e-4
    lr_decay = 0.96
    opt = "RMSprop"

    past_time = 11

    # raw_training(tile_size, sample_time, nb_epoch, exp, time_steps, batch_size, lr, lr_decay, opt, past_time)
    
    ##### GRID SEARCH
    # for batch_size in [16, 32, 50, 64]:
    #     for lr in [1e-1, 1e-2, 1e-4]:
    #         for opt in ["Adam", "RMSprop"]:
    #             for lr_decay in [0.96, 0.5]:

    for tile_size in [1500, 1000]:
        # for sample_time in ["30min", "60min", "120min", "15min"]:
        for sample_time in ["30min", "60min"]:
            time_steps = 60/float(sample_time.split("min")[0])
            raw_training(tile_size, sample_time, nb_epoch, exp, time_steps, batch_size, lr, lr_decay, opt, past_time)