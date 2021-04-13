import pandas as pd
import mlflow
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader.data_utils import seq_gen
from utils.math_utils import z_score, z_inverse
from dataset import Dataset
import torch.nn as nn
from utils import get_normalized_adj
from stgcn import STGCN
from sklearn.metrics import mean_squared_error
from skmob.tessellation import tilers
from operator import itemgetter
import skmob

def raw_training(tile_size, sample_time, nb_epoch, exp, time_steps, batch_size, lr, lr_decay, opt, past_time=11):

    df = pd.read_csv("data/df_grouped_tile"+str(tile_size)+"freq"+sample_time+".csv")
    min_tile_id = df['tile_ID_origin'].min()

    df['tile_ID_origin'] -= df['tile_ID_origin'].min()
    df['tile_ID_destination'] -= df['tile_ID_destination'].min()

    time = set()
    time_orgin_dest = []
    x_axis = int(df['tile_ID_origin'].max())+1
    y_axis = int(df['tile_ID_destination'].max())+1
    origin_dest = np.zeros([len(df['starttime'].unique()), x_axis, y_axis])
    print("Shape origin Destination matrix: ",origin_dest.shape)
    t = -1

    for el in df.itertuples():
        if(el.starttime not in time):
            time.add(el.starttime)
            t += 1
        origin_dest[t, el.tile_ID_origin, el.tile_ID_destination] = el.flow


    X = origin_dest

    day_slot = int(24*time_steps)
    print("Len of day slot is: ", day_slot)
    n_test = 10
    n_train, n_val = int((X.shape[0]-n_test*day_slot)/day_slot*0.8), int((X.shape[0]-n_test*day_slot)/day_slot*0.2)

    n_his, n_pred = past_time, 1
    n_frame = n_his+n_pred
    n_route = X.shape[1]
    C_0 = X.shape[2]

    print("Dataset is divided in ", n_train, " training ", n_val, " validation ", n_test, " test")
    print("Total ", X.shape[0]/day_slot, " / ", n_train+n_val+n_test)
    seq_train = seq_gen(n_train, X, 0, n_frame, n_route, day_slot, C_0)
    seq_val = seq_gen(n_val, X, n_train, n_frame, n_route, day_slot, C_0)
    seq_test = seq_gen(n_test, X, n_train + n_val, n_frame, n_route, day_slot, C_0)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    # Normalising data
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    train_dataset = Dataset(np.transpose(x_train, (0, 2, 1, 3)), n_his)
    val_dataset = Dataset(np.transpose(x_val, (0, 2, 1, 3)), n_his)
    test_dataset = Dataset(np.transpose(x_test, (0, 2, 1, 3)), n_his)


    params = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 4}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    training_generator = torch.utils.data.DataLoader(train_dataset, **params)
    validation_generator = torch.utils.data.DataLoader(val_dataset, **params)

    A_wave = origin_dest.sum(axis=0)
    A_wave = get_normalized_adj(A_wave)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=device, dtype=torch.float)

    net = STGCN(A_wave.shape[0],
                train_dataset.shape()[3],
                n_his,
                n_pred).to(device=torch.device('cuda'), dtype=torch.float)
    if opt == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=0.5)

    # decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    loss_criterion = nn.MSELoss()

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
        # permutation = torch.randperm(training_input.shape[0])

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



    net.train()
    # Loop over epochs
    for epoch in range(nb_epoch):
        loss = train_epoch(training_generator,
                        batch_size=batch_size)
        training_losses.append(loss)
        print('[Epoch %4d/%4d] Loss: % 2.2e' % (epoch + 1, nb_epoch, loss))

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

    print("Training loss: {}".format(training_losses[-1]))
    print("Validation loss: {}".format(validation_losses[-1]))
    print("Validation MAE: {}".format(validation_maes[-1]))
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    # plt.show()

    tot, tot_act = [], []
    for val, act in validation_generator:
        pred = net(A_wave, val.to(device, dtype=torch.float)).to(device="cpu").detach().numpy()
        tot_act.append(act.detach().numpy())
        tot.append(pred)
    tot_pred = np.concatenate(tot, axis=0)
    tot_act = np.concatenate(tot_act, axis=0)


    plt.scatter(z_inverse(tot_pred[:,:,0,:], x_stats['mean'], x_stats['std']), z_inverse(tot_act, x_stats['mean'], x_stats['std']))

    np.corrcoef(
        z_inverse(pred[:,:,0,:], x_stats['mean'], x_stats['std']).flatten(), 
        z_inverse(act, x_stats['mean'], x_stats['std']).flatten())



    # Test the model
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    local_test = []
    test_loss = []
    test_loss_matrix = []
    local_test_matrix = []
    pred_test, act_test = [], []

    print("Testing on ", x_test.shape[0], " data")

    for local_batch, local_labels in test_generator:
        net.eval()
        local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

        out = net(A_wave, local_batch)
        test_loss = loss_criterion(out, local_labels).to(device="cpu")
        local_test.append(test_loss.detach().numpy().item())
        
        # Collecting data to perform evaluation for crowd flow prediction problem
        pred_test.append(out.to(device="cpu").detach().numpy())
        act_test.append(local_labels.to(device="cpu").detach().numpy())


    # Evalute the model on training set for crowd flow prediction problem
    pred_train, act_train = [], []
    for local_batch, local_labels in training_generator:
        net.eval()
        local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

        out = net(A_wave, local_batch)
        
        # Collecting data to perform evaluation for crowd flow prediction problem
        pred_train.append(out.to(device="cpu").detach().numpy())
        act_train.append(local_labels.to(device="cpu").detach().numpy())


    mse = np.mean(local_test)
    mse_matrix = np.mean(local_test_matrix)

    out = None
    local_batch = local_batch.to(device="cpu")

    print("Test RMSE: {}".format(np.sqrt(mse)))
    print("Test MAE: {}".format(mae))

    pred_test = np.concatenate(pred_test, axis=0)
    act_test = np.concatenate(act_test, axis=0)
    
    pred_train = np.concatenate(pred_train, axis=0)
    act_train = np.concatenate(act_train, axis=0)


    tessellation = tilers.tiler.get("squared", base_shape="Manhattan, New York City, USA", meters=tile_size)
    skmob.utils.plot.plot_gdf(tessellation, popup_features=['tile_ID', 'positions'])

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


    matrix_mapping = {el[0]:el[1] for el in zip(tessellation['tile_ID'], tessellation['positions'])}


    y_max = np.array(list(tessellation['positions']))[:,1].max()+1
    x_max = np.array(list(tessellation['positions']))[:,0].max()+1


    def OD_matrix_to_map(OD_matrix, mapping, offset, map_shape):
        map_matrix = np.zeros(map_shape)
        for i in range(OD_matrix.shape[1]): # origin
            for j in range(OD_matrix.shape[3]): # destination
                x, y = mapping[str(i+offset)]
                map_matrix[:, x, y, 0, :] += OD_matrix[:, i, :, j]#.numpy() # Outflow
                x, y = mapping[str(j+offset)]
                map_matrix[:, x, y, 1, :] += OD_matrix[:, i, :, j]#.numpy() # Inflow
        return map_matrix


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

        return X_dataset, (~(X_sum[i]==0).all(1), ~(X_sum[i].T==0).all(1))


    act_test_not_norm = z_inverse(act_test, x_stats['mean'], x_stats['std']).astype(int)
    pred_test_not_norm = z_inverse(pred_test, x_stats['mean'], x_stats['std']).astype(int)
    
    act_train_not_norm = z_inverse(act_train, x_stats['mean'], x_stats['std']).astype(int)
    pred_train_not_norm = z_inverse(pred_train, x_stats['mean'], x_stats['std']).astype(int)

    A = OD_matrix_to_map(act_test_not_norm, matrix_mapping, min_tile_id, [act_test_not_norm.shape[0], x_max, y_max, 2, 1])
    P = OD_matrix_to_map(pred_test_not_norm, matrix_mapping, min_tile_id, [pred_test_not_norm.shape[0], x_max, y_max, 2, 1])

    A_train = OD_matrix_to_map(act_train_not_norm, matrix_mapping, min_tile_id, [act_train_not_norm.shape[0], x_max, y_max, 2, 1])
    P_train = OD_matrix_to_map(pred_train_not_norm, matrix_mapping, min_tile_id, [pred_train_not_norm.shape[0], x_max, y_max, 2, 1])


    A, non_empty_shape = remove_empty_rows(A[:,:,:,:,0], 2)
    P = P[:, non_empty_shape[0], :, :]
    P = P[:, :, non_empty_shape[1], :, :]
    P = P[:, :, :, :, 0]

    A_norm = z_score(A, x_stats['mean'], x_stats['std'])
    P_norm = z_score(P, x_stats['mean'], x_stats['std'])

    A_train, non_empty_shape = remove_empty_rows(A_train[:,:,:,:,0], 2)
    P_train = P_train[:, non_empty_shape[0], :, :]
    P_train = P_train[:, :, non_empty_shape[1], :, :]
    P_train = P_train[:, :, :, :, 0]

    A_norm_train = z_score(A_train, x_stats['mean'], x_stats['std'])
    P_norm_train = z_score(P_train, x_stats['mean'], x_stats['std'])


    # rmse_test = np.sqrt(mean_squared_error(A_norm.flatten(), P_norm.flatten()))


    class MinMaxNormalization(object):
        '''MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
        '''

        def __init__(self):
            pass

        def fit(self, X):
            self._min = X.min()
            self._max = X.max()
            print("min:", self._min, "max:", self._max)

        def transform(self, X):
            X = 1. * (X - self._min) / (self._max - self._min)
            X = X * 2. - 1.
            return X

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

        def inverse_transform(self, X):
            X = (X + 1.) / 2.
            X = 1. * X * (self._max - self._min) + self._min
            return X


    mmn = MinMaxNormalization()
    mmn.fit(A)

    rmse_test = np.sqrt(mean_squared_error(mmn.transform(A.flatten()), mmn.transform(P.flatten())))

    corr_test = np.corrcoef(A.flatten(), P.flatten())

    mmn_train = MinMaxNormalization()
    mmn_train.fit(A_train)

    rmse_train = np.sqrt(mean_squared_error(mmn_train.transform(A_train.flatten()), mmn_train.transform(P_train.flatten())))

    corr_train = np.corrcoef(A_train.flatten(), P_train.flatten())


    fig = plt.figure()
    plt.scatter(
        P, 
        A)
    plt.xlabel("Crowd Flow Predicted")
    plt.ylabel("Crowd Flow Real")
    fig.savefig("scatter_plot.png")


    flow = 0
    f = plt.figure(figsize=(18,8))
    ax = f.add_subplot(1,1,1)
    plt.ylabel('latitude',fontsize=20)
    plt.xlabel('longitude',fontsize=20)
    plt.yticks(fontsize=14)
    plt.title("Predicted outflow heatmap bike NYC dataset", fontsize=22)
    heatmap = A[0, :, :, flow].copy()
    for img in A[1:]:
        heatmap += img[:, :, flow]
    plt.imshow(heatmap.T, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.savefig("actual_heatmap_outflow.png")

    flow = 1
    f = plt.figure(figsize=(18,8))
    ax = f.add_subplot(1,1,1)
    plt.ylabel('latitude',fontsize=20)
    plt.xlabel('longitude',fontsize=20)
    plt.yticks(fontsize=14)
    plt.title("Predicted inflow heatmap bike NYC dataset", fontsize=22)
    heatmap = A[0, :, :, flow].copy()
    for img in A[1:]:
        heatmap += img[:, :, flow]
    plt.imshow(heatmap.T, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.savefig("actual_heatmap_inflow.png")


    flow = 0
    f = plt.figure(figsize=(18,8))
    ax = f.add_subplot(1,1,1)
    plt.ylabel('latitude',fontsize=20)
    plt.xlabel('longitude',fontsize=20)
    plt.yticks(fontsize=14)
    plt.title("Predicted outflow heatmap bike NYC dataset", fontsize=22)
    heatmap = P[0, :, :, flow].copy()
    for img in P[1:]:
        heatmap += img[:, :, flow]
    plt.imshow(heatmap.T, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.savefig("predicted_heatmap_outflow.png")

    flow = 1
    f = plt.figure(figsize=(18,8))
    ax = f.add_subplot(1,1,1)
    plt.ylabel('latitude',fontsize=20)
    plt.xlabel('longitude',fontsize=20)
    plt.yticks(fontsize=14)
    plt.title("Predicted inflow heatmap bike NYC dataset", fontsize=22)
    heatmap = P[0, :, :, flow].copy()
    for img in P[1:]:
        heatmap += img[:, :, flow]
    plt.imshow(heatmap.T, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.savefig("predicted_heatmap_inflow.png")


    params = {
        "tile_size" : tile_size,
        "sample_time" : sample_time,
        "batch_size" : batch_size,
        "lr" : lr,
        "epochs" : nb_epoch,
        "lr_decay": lr_decay, 
        "optimizer": opt
    }
    metrics = {
        "train RMSE" : rmse_train,
        "train corr" : corr_train[0][1],
        "train MAE" : validation_maes[-1],
        "test RMSE" : rmse_test,
        "test corr" : corr_test[0][1]
        }

    
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
    

    mlflow.log_artifact("predicted_heatmap_inflow.png")
    mlflow.log_artifact("actual_heatmap_inflow.png")
    mlflow.log_artifact("predicted_heatmap_outflow.png")
    mlflow.log_artifact("actual_heatmap_outflow.png")
    mlflow.log_artifact("scatter_plot.png")

    # Ending MLFlow run
    mlflow.end_run()
