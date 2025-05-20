# StoxLSTM
StoxLSTM: A Stochastic xLSTM for Probabilistic Time Series Forecasting

1. Install requirments. ```conda env create -n StoxLSTM -f requirements.yaml```

2. Training. All the scripts are in the directory ```./scripts```. For example, if you want to get the multivariate forecasting results for ILI dataset, just run the following command, and you can open ```./training_info``` and ```./figure``` to see the results once the training is done:
```
bash ./script/ili.sh
```

You can adjust the hyperparameters based on your needs.