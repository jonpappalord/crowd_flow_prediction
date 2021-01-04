import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import pickle
import sys
import time

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skmob.tessellation import tilers

sys.path.append("/mnt/d/ASE/Thesis/Project/CrowdFlowPrediction")

from utils.load_datasets import BikeNYC
import utils.metrics as metrics
from utils.postprocessing import print_heatmap, nrmse_quantile
from utils.read_csv import csv_stub
from models.deepst.STResNet import stresnet
np.random.seed(1337)  # for reproducibility

tile_size = 1500
sample_time = "60min"

nb_epoch = 500  # number of epoch at training stage
batch_size = 32  # batch size
T = 24  # number of time intervals at a day
lr = 0.0002  # learning rate
len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence

nb_residual_unit = 4  # number of residual units

nb_flow = 2  # there are two types of flows: inflow and outflow
days_test = 10
len_test = T * days_test


path_result = 'RET'
path_model = 'MODEL'

experiment_name = " ".join(["bike","NN"])

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

def build_model(external_dim, map_height=16, map_width=8):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model

def make_report(params, metrics, experiment_id):
    """
        Create a mlflow report

        Parameters
        ----------
        - params: a dict of the hyperparameters
        - experiment_id: an identificator for the current experiment
    """
    # Starting MLFlow run
    mlflow.start_run(run_name=str(params), experiment_id=experiment_id)
    
    print("Uri: " + str(mlflow.get_artifact_uri()))

    mlflow.log_params(params)

    # Logging model metrics
    mlflow.log_metrics(metrics)

    # Ending MLFlow run
    mlflow.end_run()

def save_corr_fig(data, data_pred,figname):
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(1,1,1)
    plt.ylabel('Predicted flow',fontsize=20)
    plt.xlabel('Real flow',fontsize=20)
    plt.title("Compare real flows against predicted in bike NYC dataset", fontsize=22)
    start_date = 4
    sampling = 24
    flow = 1
    plt.scatter(
        # data_pred[start_date:start_date+7*sampling,:,:,flow].flatten(),
        # data[start_date:start_date+7*sampling,:,:,flow].flatten())
        data[:,:,:,flow],
        data_pred[:,:,:,flow])
    # plt.loglog()
    x = np.logspace(0, np.log10(np.max(data)))
    plt.plot(x, x, '--k')
    plt.show()
    plt.savefig(figname+".png")

def main():
    # Setting MLFlow
    mlflow.set_experiment(experiment_name = experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    sample_time = "60min"
    tile_sizes = [500, 1000, 1500, 2000, 3000]

    for tile_size in tile_sizes:
        # load data
        print("loading data...")
        ts = time.time()
        X_dataset, time_string = csv_stub(tile_size, sample_time)
        
        preprocessing_file = "preprocessing.NYC"+str(tile_size)+sample_time+".pkl"
        
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
                T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
                preprocess_name=preprocessing_file, meta_data=True, stdata=(X_dataset, time_string))
        
        print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
        print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

        print("\ncompiling model...")
        ts = time.time()
        model = build_model(external_dim, X_dataset.shape[1], X_dataset.shape[2])

        hyperparams_name = 'Tile{}.freq{}.c{}.p{}.t{}.resunit{}.lr{}'.format(
            tile_size, sample_time, len_closeness, len_period, len_trend, nb_residual_unit, lr)
        fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

        early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
        model_checkpoint = ModelCheckpoint(
            fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

        print("\nelapsed time (compiling model): %.3f seconds\n" %
            (time.time() - ts))

        print("training model...")
        ts = time.time()
        history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[early_stopping, model_checkpoint],
                            verbose=2)
        model.save_weights(os.path.join(
            'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
        pickle.dump((history.history), open(os.path.join(
            path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
        print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

        print('evaluating using the model that has the best loss on the valid set')
        ts = time.time()
        model.load_weights(fname_param)
        score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                            0] // 48, verbose=0)
        print('Train score: %.6f Train rmse: %.6f %.6f' %
            (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
        test_score = model.evaluate(
            X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
        print('Test score: %.6f Test rmse: %.6f %.6f' %
            (test_score[0], test_score[1], test_score[1] * (mmn._max - mmn._min) / 2.))
        print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

        params = {
            "tile_size": tile_size,
            "sample_time":sample_time,
            # "batch_size":batch_size,
            # "lr":lr,
            # "nb_residual_unit":nb_residual_unit
        }
        metrics = {
            "train RMSE": score[1] * (mmn._max - mmn._min) / 2.,
            "test RMSE" : test_score[1] * (mmn._max - mmn._min) / 2.
            }

        make_report(params, metrics, exp)

        try:
            Y_pred = model.predict(X_test)

            data = mmn.inverse_transform(Y_test)
            data_pred = np.array(mmn.inverse_transform(Y_pred), dtype=int)
            save_corr_fig(data, data_pred, hyperparams_name)
        except:
            pass


if __name__ == '__main__':

    main()
