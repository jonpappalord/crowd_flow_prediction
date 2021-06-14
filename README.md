## Crowd Flow prediction - AdjNet in Pytorch

An implementation of the Adjacency Matrix Neural Network (AdjNet) used to solve flow prediction problem.


# Install
At first you need to clone this repository:
```shell
$ git clone https://github.com/jonpappalord/crowd_flow_prediction
$ cd crowd_flow_prediction
```

Create a new environment:
```shell
$ python -m venv yourenvname 
$ source yourenvname/bin/activate
```


Launch the following command to install the required packages

```shell
$ pip install -r requirements.txt
```

# Usage

We have prepared an instance for running the model, that is `test_model.py`.

If you want to try them on the bike NYC bike dataset you can download it from the [official page](https://www.citibikenyc.com/system-data) and save them in the `data/BikeNYC` folder. Then just tune the hyperparameters as you wish in the `test_model.py` and launch it with

```shell
$ python test_model.py
```