import matplotlib.pyplot as plt
import numpy as np
import pickle

from deepst.preprocessing import MinMaxNormalization


def load_obj(file_path):
    with open(file_path + '.pkl', 'rb') as f:
        return pickle.load(f)


def print_heatmap(data, flow):
    """
        flow: inflow or outlow heatmap
    """
    heatmap = data[0, :, :, flow]
    np.shape(heatmap)
    for img in data[1:]:
        # print(np.shape(i[:, :, 0]))
        heatmap += img[:, :, flow]
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.show()

def print_weekly_plot(data, start_date, location, flow=0):
    """
        data: inflow or outflow crowd movement
        start_date: an integer representing the first day of the week
        location: a tuple containing x and y coordinates
    """
    # Plot office area from 14/03/2016 to 21/03/2016 
    # Example of office area is (8,3), residential area (30,24)
    start_date *= 48
    plt.plot(list(range(start_date)), data[start_date:start_date+7*48,location[0],location[1],flow])

def print_correlation_matrix(data, data_pred, start_date, location, flow=0):
    plt.scatter(
        data_pred[start_date:start_date+7*48,location[0],location[1],flow], 
        data[start_date:start_date+7*48,location[0],location[1],flow], marker='s')


def print_plots(data, data_pred):
    mmn = load_obj("preprocessing")

    data = mmn.inverse_transform(data)
    data_pred = mmn.inverse_transform(data_pred)

    print_heatmap(data, 0)

    print_weekly_plot(data, 4, (8,3))

    print_correlation_matrix(data, data_pred, 4, (8,3))
    
