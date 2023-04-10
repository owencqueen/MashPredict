# Deep Learning for Unaligned Geolocation of Poplar Trees
Predicting locations of poplar trees from whole genome sequences using Mash sketches, accepted to AI for Science Workshop at NeurIPS 2022. You can find the paper here: https://arxiv.org/abs/2301.13387.

<img src="https://github.com/owencqueen/MashPredict/blob/main/maps.png" width="500" height="500">

## Running code

All experiments are facilitated through Python scripts:

```
# ElasticNet and XGBoost (aligned and Mash-encoded)
>>> cd onehot_models
>>> python3 multiple_ML.py [-options]

# KNN
>>> cd dist_models
>>> python3 multiple_knn.py [-options]

# MashNet
>>> cd nn_models
>>> python3 mashnet_cv.py [-options]
```

## Data Availability

All SRA codes are available in the [SRA_keys.txt](https://github.com/owencqueen/MashPredict/blob/main/SRA_keys.txt). Precomputed Mash sketches at the sketch sizes specified in the paper are provided in the [`data`](https://github.com/owencqueen/MashPredict/tree/main/data) directory within the GitHub as tar files.
