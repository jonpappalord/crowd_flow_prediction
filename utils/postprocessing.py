import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from utils.config import Config
from utils.preprocessing_functions import MinMaxNormalization
from sklearn.metrics import mean_squared_error

RESULTDIR = "results/"+str(int(datetime.datetime.now().timestamp()))
if os.path.isdir("results") is False:
    os.mkdir("results")
    if os.path.isdir(RESULTDIR) is False:
        os.mkdir(RESULTDIR)


def load_obj(file_path):
    with open(file_path + '.pkl', 'rb') as f:
        return pickle.load(f)


def print_heatmap(data, flow):
    """
        flow: inflow or outlow heatmap
    """
    heatmap = data[0, :, :, flow]
    for img in data[1:]:
        heatmap += img[:, :, flow]
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.show()

def print_weekly_plot(data, start_date, location, flow=0, sampling=24):
    """
        data: inflow or outflow crowd movement
        start_date: an integer representing the first day of the week
        location: a tuple containing x and y coordinates
    """
    # Plot office area from 14/03/2016 to 21/03/2016 
    # Example of office area is (8,3), residential area (30,24)
    start_date *= sampling
    plt.plot(list(range(7*sampling)), data[start_date:start_date+7*sampling,location[0],location[1],flow])

def print_correlation_matrix(data, data_pred, start_date, location, flow=1, sampling=48):
    fig = plt.figure()
    plt.scatter(
        data_pred[start_date:start_date+7*sampling,location[0],location[1],flow], 
        data[start_date:start_date+7*sampling,location[0],location[1],flow], marker='s')
    plt.xlabel("Crowd Flow Predicted")
    plt.ylabel("Crowd Flow Attended")
    fig.savefig(RESULTDIR+"/scatter_matrix.png")

def rmse_std(predictions, targets):
    return np.std(np.sqrt(((predictions - targets) ** 2)))

def nrmse_quantile(data, data_pred, quantile, start_date, n_days=1, flow=1, sampling=24):
    start_date *=sampling
    n_days *= sampling
    end_date = start_date+n_days
    
    data = np.array([data[i:i+sampling].mean(axis=0) for i in range(start_date, end_date, sampling)]).flatten()
    data_pred = np.array([data_pred[i:i+sampling].mean(axis=0) for i in range(start_date, end_date, sampling)]).flatten()
    
    # Permuting data and applying the same permutation to the prediction
    p = data.argsort()
    data = data[p]
    data_pred = data_pred[p]

    decili = [d for d in np.array_split(data, quantile)]
    decili_pred = [d for d in np.array_split(data_pred, quantile)]
    
    for i in range(quantile):
        plt.hist(decili[i])
        plt.savefig("decili_distr"+str(i))
        plt.clf()

    rmse_decili = np.array([(mean_squared_error(decili[i], decili_pred[i], squared=False),
               rmse_std(decili[i], decili_pred[i]))
               for i in range(quantile)])

    fig = plt.figure()
    plt.errorbar(list(range(quantile)), rmse_decili[:,0], rmse_decili[:,1])
    plt.xlabel("Decile ")
    plt.ylabel("NRMSE")
    fig.savefig(RESULTDIR+"/nrmse.png")
    # fig.savefig("results/nrmse.png")



def print_plots(data, data_pred):
    mmn = load_obj("preprocessing")

    data = mmn.inverse_transform(data)
    data_pred = mmn.inverse_transform(data_pred)

    print_heatmap(data, 0)

    print_weekly_plot(data, 4, (8,3))

    print_correlation_matrix(data, data_pred, 4, (8,3))
    
