## AdjNet: a deep learning approach for Crowd Flow Prediction

Python impletmentaion of the thesis ***AdjNet: a deep learning approach for Crowd Flow Prediction*** ([Link](https://etd.adm.unipi.it/t/etd-06092021-221917/))

### Introduction

We propose AdjNet (Adjacency Matrix Neural Network), which solves the Crowd Flow Prediction problem using an approach based on Graph Convolutional Networks (GCN) and Convolutional Neural Networks (CNN). In the first stage, we first represent the area taken into account using different tessellations. In the second stage, we train our model using the New York City Bike Share dataset to predict flows among regions.

<img src = "https://github.com/jonpappalord/crowd_flow_prediction/blob/main/Figure/AdjNetArch.png?raw=1" style="width: 75%;"/>
<em> Architecture of CrowdNet. </em>

### Result

We evaluate our model using different tessellation, evaluating them exploiting the RMSE metric. We compare it using as baseline another Deep Learning approach named STResNet ([Link](https://arxiv.org/abs/1610.00081)).

**RMSE**

<table class="tg">
<thead>
  <tr>
    <th class="tg-gvcd"></th>
    <th class="tg-gvcd"></th>
    <th class="tg-v0nz" colspan="6">Tile Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-gvcd"></td>
    <td class="tg-gvcd"></td>
    <td class="tg-v0nz" colspan="2">750m</td>
    <td class="tg-v0nz" colspan="2">1000m<br></td>
    <td class="tg-v0nz" colspan="2">1500m</td>
  </tr>
  <tr>
    <td class="tg-gvcd"></td>
    <td class="tg-gvcd"></td>
    <td class="tg-v0nz">AdjNet</td>
    <td class="tg-v0nz">STResNet</td>
    <td class="tg-v0nz">AdjNet</td>
    <td class="tg-v0nz">STResNet</td>
    <td class="tg-v0nz">AdjNet</td>
    <td class="tg-v0nz">STResNet</td>
  </tr>
  <tr>
    <td class="tg-gvcd" rowspan="4">Time intervals</td>
    <td class="tg-8c31">15min</td>
    <td class="tg-anz3">1.71</td>
    <td class="tg-anz3">1.69</td>
    <td class="tg-anz3">2.76</td>
    <td class="tg-anz3">2.35</td>
    <td class="tg-anz3">3.73</td>
    <td class="tg-anz3">3.35</td>
  </tr>
  <tr>
    <td class="tg-8c31">30min</td>
    <td class="tg-anz3">3.69</td>
    <td class="tg-anz3">2.65</td>
    <td class="tg-anz3">5.23</td>
    <td class="tg-anz3">4.85</td>
    <td class="tg-anz3">8.93</td>
    <td class="tg-anz3">5.64</td>
  </tr>
  <tr>
    <td class="tg-8c31">45min</td>
    <td class="tg-anz3">4.34</td>
    <td class="tg-anz3">3.67</td>
    <td class="tg-anz3">6.68</td>
    <td class="tg-anz3">5.63</td>
    <td class="tg-anz3">11.3</td>
    <td class="tg-anz3">10.91</td>
  </tr>
  <tr>
    <td class="tg-8c31">60min</td>
    <td class="tg-anz3">5.18</td>
    <td class="tg-anz3">5.44</td>
    <td class="tg-anz3">8.53</td>
    <td class="tg-anz3">9.38</td>
    <td class="tg-anz3">11.1</td>
    <td class="tg-anz3">11.66</td>
  </tr>
</tbody>
</table>
<em>  Performance of the ST-ResNet and CrowdNet models for crowd flow prediction problem on the BikeNYC dataset. </em>
<br />


**Images**


A visual comparison of the mean real crowd inflows with the mean crowd inflows predicted by STResNet and AdjNet is provided by the following image.


<img src = "https://github.com/jonpappalord/crowd_flow_prediction/blob/main/Figure/Heatmap_Inflow_1000m_30min.png?raw=1" style="width: 75%;"/>

<em> Comparison of mean real crowd inflows (center) with those predicted by CrowdNet (left) and STResNet (right) using tile size of 1000m and time intervals of 30 minutes.</em>


### Train

#### Dataset

Flow data [available online](https://www.citibikenyc.com/system-data)



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

We have prepared an instance for running the model, that is `main.py`.

<!-- If you want to try them on the bike NYC bike dataset you can download it from the [official page](https://www.citibikenyc.com/system-data) and save them in the `data/BikeNYC` folder. Then just tune the hyperparameters as you wish in the `main.py` and launch it with -->

```shell
$ python main.py
```
