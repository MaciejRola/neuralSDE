import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import pandas as pd
import os.path
from neuralSDE.Utilities.MC import SABR_MC
from neuralSDE.Models.NeuralLV import NeuralLV
from neuralSDE.Wasserstein.Wasserstein import train_Wasserstein


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = 'cpu'
print(f'Using {device}')

# data for parametrization of neuralLV model
options = pd.read_csv('./neuralSDE/StandardApproach/Data/Options_results.csv')
options = options[options['Expiration_date'].isin([0.5, 1.0, 1.5, 2.0])]
maturities = options['Expiration_date'].unique()
strikes = options['Strike'].unique()

# market parameter
F0 = 1
Time_horizon = 2
rfr = 0.05
S0 = F0 * np.exp(-rfr * Time_horizon)

# model parameters
num_layers = 4
num_layers_hedging = 3
layer_size = 50
layer_size_hedging = 30
dropout = 0.0
n_maturities = maturities.shape[0]
n_strikes = strikes.shape[0]
use_hedging = True
use_batchnorm = False

# simulation parameters
batch_size = 40000
epochs = 100000
N_simulations = 20 * batch_size
N_steps = 96
period_length = N_steps // n_maturities

# Monte Carlo test data
MC_samples_test = 200000
test_normal_variables = torch.randn(MC_samples_test, N_steps, requires_grad=False)
# We will use antithetic Brownian paths for testing
test_normal_variables = torch.cat([test_normal_variables, -test_normal_variables], 0)

# parameters for the SABR model
sigma_0 = 0.3  # initial volatility of the underlying
alpha = 0.2  # volatility of future price volatility
beta = 0.6  # exponent in SDE
rho = 0.2  # correlation coefficient

if ~os.path.exists('/neuralSDE/Wasserstein/Data/target_Wasserstein.pth.tar'):
    sabr = SABR_MC(F_0=F0, sigma_0=sigma_0, r=rfr, alpha=alpha, beta=beta, rho=rho, Time_horizon=Time_horizon, N_steps=N_steps, N_simulations=N_simulations)
    target_numpy = sabr.simulate_paths()
    target = torch.tensor(target_numpy, dtype=torch.float32)
    torch.save(target, '/neuralSDE/Wasserstein/Data/target_Wasserstein.pth.tar')
else:
    target = torch.load('/neuralSDE/Wasserstein/Data/target_Wasserstein.pth.tar')

with open("/neuralSDE/Wasserstein/Results/log_eval_LV_Wasserstein.txt", "w") as f:
    f.write('epoch,loss\n')

modelLV = NeuralLV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                   N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                   S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                   num_layers=num_layers, layer_size=layer_size,
                   num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                   test_normal_variables=test_normal_variables)

print('Neural Local Volatility Model initiated')
#best_modelLV = torch.load('../StandardApproach/Results/NeuralLV.pth.tar')
#modelLV.load_state_dict(best_modelLV['state_dict'])
#print('Neural Local Volatility Model loaded')
train_Wasserstein(modelLV, target, epochs, batch_size)
print('Neural Local Volatility Model trained')
