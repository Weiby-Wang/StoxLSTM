# StoxLSTM
StoxLSTM: A Stochastic xLSTM for Probabilistic Time Series Forecasting

1. Install requirments. ```conda env create -n StoxLSTM -f requirements.yaml```

2. Training. All the scripts are in the directory ```./scripts```. For example, if you want to get the multivariate forecasting results for ILI dataset, just run the following command, and you can open ```./training_info``` and ```./figure``` to see the results once the training is done:
```
bash ./script/ili.sh
```

You can adjust the hyperparameters based on your needs. The class Z_generation_model() in the StoxLSTM_layers.py demonstrates two methods for initializing the latent variable z0: one by sampling from a standard normal distribution, and the other by initializing directly with a constant zero vector. These two initialization approaches lead to subtle differences in training performance across different datasets. Adjusting the initialization method of z0 can help achieve improved training outcomes.