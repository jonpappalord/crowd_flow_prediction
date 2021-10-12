import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import skmob
from skmob.tessellation import tilers
from src.AdjNet.utils.config import Config
import torch
import torch.nn as nn
import sys

from src.AdjNet.utils import get_normalized_adj
from src.AdjNet.data_loader.data_utils import seq_gen, load_dataset, split_and_scale, get_matrix_mapping, restore_od_matrix_pred, to_2D_map
from src.AdjNet.stgcn import STGCN

temp_dir = "temp/"
if os.path.isdir(temp_dir) is False:
    os.mkdir(temp_dir)

path_predictions = temp_dir+"predictions/"
if os.path.isdir(path_predictions) is False:
    os.mkdir(path_predictions)


def train_and_evaluate(tile_size, sample_time, nb_epoch, exp, time_steps, batch_size, lr, lr_decay, opt, past_time=11):
    print("Loading data")
    od_matrix, empty_indices, min_tile_id = load_dataset(tile_size, sample_time)

    n_his, n_pred = past_time, 1
    train_dataset, val_dataset, test_dataset = split_and_scale(od_matrix, time_steps, past_time, n_pred)

    params = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 2}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    A_wave = od_matrix.sum(axis=0)
    A_wave = torch.from_numpy(get_normalized_adj(A_wave)).to(device=device, dtype=torch.float)

    # Building the model
    net = STGCN(A_wave.shape[0],
                train_dataset.shape()[3],
                n_his,
                n_pred).to(device=torch.device(device), dtype=torch.float)
                
    if opt == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=0.5)

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
    loss_criterion = nn.MSELoss()

    # Training the model
    training_losses = []
    validation_losses = []
    validation_maes = []

    def train_epoch(training_generator, batch_size):
        """
        Trains one epoch with the given data.
        :param training_input: Training inputs of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param training_target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """

        epoch_training_losses = []
        # Training
        for local_batch, local_labels in training_generator:
            net.train()
            optimizer.zero_grad()

            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

            out = net(A_wave, local_batch)
    #         print("out shape: ", out.shape)
    #         print("local_batch shape: ", local_labels.shape)
            loss = loss_criterion(out, local_labels)
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        return sum(epoch_training_losses)/len(epoch_training_losses)

    training_generator = torch.utils.data.DataLoader(train_dataset, **params)
    validation_generator = torch.utils.data.DataLoader(val_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    net.train()
    # Loop over epochs
    for epoch in range(nb_epoch):
        loss = train_epoch(training_generator,
                        batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.set_grad_enabled(False):
            local_val = []
            for local_batch, local_labels in validation_generator:
                net.eval()
                local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

                out = net(A_wave, local_batch)
                val_loss = loss_criterion(out, local_labels).to(device="cpu")
                local_val.append(np.asscalar(val_loss.detach().numpy()))

            validation_losses.append(np.mean(local_val))
            std_val, mean_val = val_dataset.get_stats()['std'], val_dataset.get_stats()['mean']
            out_unnormalized = out.detach().cpu().numpy() #* std_val + mean_val #*stds[0]+means[0]
            target_unnormalized = local_labels.detach().cpu().numpy() #* std_val + mean_val #*stds[0]+means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)

            out = None
            local_batch = local_batch.to(device="cpu")
            local_labels = local_labels.to(device="cpu")
        
        my_lr_scheduler.step()
        print('[Epoch %4d/%4d] Loss: % 2.2e Val Loss: % 2.2e' % (epoch + 1, nb_epoch, loss, np.mean(local_val)))

    print("Training loss: {}".format(training_losses[-1]))
    print("Validation loss: {}".format(validation_losses[-1]))
    print("Validation MAE: {}".format(validation_maes[-1]))

    torch.save(net, Config().DATAPATH+"/Models/AdjNet_"+str(tile_size)+"m_"+sample_time+"min.pt")


    ### TEST DATASET ###
    local_test = []
    # test_loss = []
    pred_test, act_test = [], []
    for local_batch, local_labels in test_generator:
        net.eval()
        local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

        out = net(A_wave, local_batch)
        test_loss = loss_criterion(out, local_labels).to(device="cpu")
        local_test.append(test_loss.detach().numpy().item())
        
        # Collecting data to perform evaluation for crowd flow prediction problem
        pred_test.append(out.to(device="cpu").detach().numpy())
        act_test.append(local_labels.to(device="cpu").detach().numpy())
    pred_test = np.concatenate(pred_test, axis=0)
    act_test = np.concatenate(act_test, axis=0)

    
    # Evalute the model on training set for crowd flow prediction problem
    pred_train, act_train = [], []
    for local_batch, local_labels in training_generator:
        net.eval()
        local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

        out = net(A_wave, local_batch)
        
        # Collecting data to perform evaluation for crowd flow prediction problem
        pred_train.append(out.to(device="cpu").detach().numpy())
        act_train.append(local_labels.to(device="cpu").detach().numpy())
    pred_train = np.concatenate(pred_train, axis=0)
    act_train = np.concatenate(act_train, axis=0)
    
    act_test = restore_od_matrix_pred(act_test, empty_indices)
    pred_test = restore_od_matrix_pred(pred_test, empty_indices)

    # Save predictions to apply different evaluation metrics
    with open(path_predictions+'real_flow_tile'+str(tile_size)+"time_interval"+sample_time+'.pkl', 'wb') as handle:
        pickle.dump(act_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_predictions+'predicted_flow_tile'+str(tile_size)+"time_interval"+sample_time+'.pkl', 'wb') as handle:
        pickle.dump(pred_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##### TRAIN EVALUATION FOR CROWD FLOW ##########
    # Evaluating the model for crowd flow prediction task
    matrix_mapping, x_max, y_max = get_matrix_mapping(tile_size)

    # Normalising data and evaluating it
    # Rescaling data
    # pred_train = stdn.inverse_transform(pred_train.reshape(-1, 1)).reshape(pred_train.shape)
    # act_train = stdn.inverse_transform(act_train.reshape(-1, 1)).reshape(act_train.shape)
    #ATTENZIONE TEST DA TOGLIERE!! TODO

    act_train = restore_od_matrix_pred(act_train, empty_indices)
    pred_train = restore_od_matrix_pred(pred_train, empty_indices)

    actual_map, predicted_map = to_2D_map(act_train, pred_train, matrix_mapping, empty_indices, min_tile_id, x_max, y_max)

    mmn = MinMaxScaler()
    mmn.fit(actual_map.reshape(-1,1))

    rmse_train = np.sqrt(mean_squared_error(actual_map.flatten().reshape(-1,1), predicted_map.flatten().reshape(-1,1)))
    
    actual_map = mmn.transform(actual_map.flatten().reshape(-1,1)).reshape(actual_map.shape)
    predicted_map = mmn.transform(predicted_map.flatten().reshape(-1,1)).reshape(predicted_map.shape)

    nrmse_train = np.sqrt(mean_squared_error(actual_map.flatten().reshape(-1,1), predicted_map.flatten().reshape(-1,1)))
    corr_train = np.corrcoef(actual_map.flatten(), predicted_map.flatten())


    ##### TEST EVALUATION FOR CROWD FLOW ##########
    
    # Normalising data and evaluating it
    actual_map, predicted_map = to_2D_map(act_test, pred_test, matrix_mapping, empty_indices, min_tile_id, x_max, y_max)

    mmn = MinMaxScaler()
    mmn.fit(actual_map.flatten().reshape(-1,1))

    rmse_test = np.sqrt(mean_squared_error(actual_map.flatten().reshape(-1,1), predicted_map.flatten().reshape(-1,1)))

    actual_map = mmn.transform(actual_map.flatten().reshape(-1,1)).reshape(actual_map.shape)
    predicted_map = mmn.transform(predicted_map.flatten().reshape(-1,1)).reshape(predicted_map.shape)

    nrmse_test = np.sqrt(mean_squared_error(actual_map.flatten().reshape(-1,1), predicted_map.flatten().reshape(-1,1)))
    corr_test = np.corrcoef(actual_map.flatten(), predicted_map.flatten())

    # Plot images
    actual_map = mmn.inverse_transform(actual_map.flatten().reshape(-1,1)).reshape(actual_map.shape)
    predicted_map = mmn.inverse_transform(predicted_map.flatten().reshape(-1,1)).reshape(predicted_map.shape)


    params = {
        "tile_size" : tile_size,
        "sample_time" : sample_time,
        "batch_size" : batch_size,
        "lr" : lr,
        "epochs" : nb_epoch,
        "lr_decay": lr_decay, 
        "optimizer": opt,
        "len_closeness": past_time
    }
    metrics = {
        "train NRMSE" : nrmse_train,
        "train corr" : corr_train[0][1],
        "train MAE" : validation_maes[-1],
        "test NRMSE" : nrmse_test,
        "test RMSE" : rmse_test,
        "train RMSE" : rmse_train,
        "test corr" : corr_test[0][1]
        }

    print("RMSE Error train: ", rmse_train)
    print("RMSE Error test: ", rmse_test)
    print("NRMSE Error train: ", nrmse_train)
    print("NRMSE Error test: ", nrmse_test)
    print("Corr test: ", corr_test[0][1])

    # Starting MLFlow run
    mlflow.start_run(run_name=str(params), experiment_id=exp.experiment_id)
    
    print("Uri: " + str(mlflow.get_artifact_uri()))

    mlflow.log_params(params)

    # Logging model metrics
    mlflow.log_metrics(metrics)

    for epoch, tr_rmse in enumerate(training_losses):
        mlflow.log_metric(key="Training loss", value = tr_rmse, step = epoch+1)

    for epoch, val_rmse in enumerate(validation_losses):
        mlflow.log_metric(key="Validation loss", value = val_rmse, step = epoch+1)

    for epoch, val_loss in enumerate(validation_maes):
        mlflow.log_metric(key="Validation MAE", value = val_loss, step = epoch+1)
    
    # fig_name = "ts"+str(params['tile_size'])+"_f"+str(params['sample_time'])
    

    # mlflow.log_artifact("heatmap_tile"+str(tile_size)+"time_interval"+sample_time+"_inflow_predicted.png")
    # mlflow.log_artifact("heatmap_tile"+str(tile_size)+"time_interval"+sample_time+"_outflow_predicted.png")
    # mlflow.log_artifact("heatmap_tile"+str(tile_size)+"time_interval"+sample_time+"_inflow_real.png")
    # mlflow.log_artifact("heatmap_tile"+str(tile_size)+"time_interval"+sample_time+"_outflow_real.png")
    # mlflow.log_artifact("decile_tile"+str(tile_size)+"time_interval"+sample_time+".png")
    # mlflow.log_artifact("scatter_plot_tile"+str(tile_size)+"time_interval"+sample_time+".png")
    # mlflow.log_artifact("decile_tile"+str(tile_size)+"time_interval"+sample_time+".csv")

    # Ending MLFlow run
    mlflow.end_run()