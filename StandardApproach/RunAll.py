import sys
import os
sys.path.append(os.getcwd())
from NeuralLV import NeuralLV, train as trainLV
from NeuralLSV import NeuralLSV, train as trainLSV
from NeuralSDE import NeuralSDE, train as trainSDE
import torch
import numpy as np
import pandas as pd
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = 'cpu'
print(f'Using {device}')

options = pd.read_csv('./neuralSDE/StandardApproach/Data/Options_results.csv')
options = options[options['Expiration_date'].isin([0.5, 1.0, 1.5, 2.0])]
maturities = options['Expiration_date'].unique()
strikes = options['Strike'].unique() / 100
target = pd.pivot_table(options, values='Price', index='Strike', columns='Expiration_date').to_numpy() / 100
target = torch.tensor(target, requires_grad=True, dtype=torch.float, device=device)

# market parameter
F0 = 100 / 100
n_maturities = maturities.shape[0]
n_strikes = strikes.shape[0]
Time_horizon = 2
rfr = 0.05
S0 = F0 * np.exp(-rfr * Time_horizon)

# model parameters
num_layers = 4
num_layers_hedging = 3
layer_size = 50
layer_size_hedging = 30
dropout = 0.0
use_hedging = True
use_batchnorm = False

# simulation parameters
batch_size = 10000
epochs = 1000
N_simulations = 20 * batch_size
N_steps = 96
period_length = N_steps // n_maturities

# Monte Carlo test data
MC_samples_test = 200000
test_normal_variables = torch.randn(2, MC_samples_test, N_steps, requires_grad=False)
# We will use antithetic Brownian paths for testing
test_normal_variables = torch.cat([test_normal_variables, -test_normal_variables], 1)

if not os.path.exists('./neuralSDE/StandardApproach/Results/NeuralLV.pth.tar'):
    with open("./neuralSDE/StandardApproach/Results/log_eval_NeuralLV.txt", "w") as f:
        f.write('epoch,loss\n')

    modelLV = NeuralLV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                       N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                       S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                       num_layers=num_layers, layer_size=layer_size,
                       num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                       test_normal_variables=test_normal_variables[0])
    print('Neural Local Volatility Model initiated')
    trainLV(modelLV, maturities=maturities, strikes=strikes, target=target, batch_size=batch_size, epochs=epochs, threshold=2e-5)
    print('Neural Local Volatility Model trained')
    torch.cuda.empty_cache()

if not os.path.exists('./neuralSDE/StandardApproach/Results/NeuralLSV.pth.tar'):
    with open("./neuralSDE/StandardApproach/Results/log_eval_NeuralLSV.txt", "w") as f:
        f.write('epoch,loss\n')

    modelLSV = NeuralLSV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                         N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                         S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                         num_layers=num_layers, layer_size=layer_size,
                         num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                         test_normal_variables=test_normal_variables)
    print('Neural Local Stochastic Volatility Model initiated')
    trainLSV(modelLSV, maturities=maturities, strikes=strikes, target=target, batch_size=batch_size, epochs=epochs, threshold=2e-5)
    print('Neural Local Stochastic Volatility Model trained')
    torch.cuda.empty_cache()

if not os.path.exists('./neuralSDE/StandardApproach/Results/NeuralSDE.pth.tar'):
    with open("./neuralSDE/StandardApproach/Results/log_eval_NeuralSDE.txt", "w") as f:
        f.write('epoch,loss\n')

    modelSDE = NeuralSDE(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                         N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                         S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                         num_layers=num_layers, layer_size=layer_size,
                         num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                         test_normal_variables=test_normal_variables)
    print('Neural Stochastic Differential Equation Model initiated')
    trainSDE(modelSDE, maturities=maturities, strikes=strikes, target=target, batch_size=batch_size, epochs=epochs, threshold=2e-5)
    print('Neural Stochastic Differential Equation Model trained')
