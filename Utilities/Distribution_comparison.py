import numpy as np
import matplotlib.pyplot as plt
from Utilities.MC import SABR_MC
import pandas as pd
import torch
from Models.NeuralLV import NeuralLV
import seaborn as sns

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = 'cpu'
print(f'Using {device}')

# data for parametrization of neuralLV model
options = pd.read_csv('../StandardApproach/Data/Options_results.csv')
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
batch_size = 4000
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
sabr = SABR_MC(F_0=F0, sigma_0=sigma_0, r=rfr, alpha=alpha, beta=beta, rho=rho, Time_horizon=Time_horizon, N_steps=N_steps, N_simulations=N_simulations)
model = NeuralLV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                 N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                 S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                 num_layers=num_layers, layer_size=layer_size,
                 num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                 test_normal_variables=test_normal_variables)
best_modelLV = torch.load('../Wasserstein/Results/Wasserstein_NeuralLV.pth.tar')
model.load_state_dict(best_modelLV['state_dict'])
model.eval()


def compare_distributions(model, MC, N_samples=100000):
    # Generate data from the model
    with torch.no_grad():
        model.to(model.device)
        model.N_simulations = N_samples
        model_values = model.simulate_paths(N_samples)[:, -1]
        model_values = model_values.detach().cpu().numpy()
    # Generate data from the target
    MC.N_simulations = N_samples
    MC_values = MC.simulate_paths()[:, -1]
    MC_values = MC_values[MC_values != 0]
    MC_values = MC_values * np.exp(-rfr * Time_horizon)
    # Plot the data
    sns.kdeplot(data=[model_values, MC_values], fill=True, common_norm=False, legend=True)
    plt.show()
    plt.close()
    # plt.hist(model_values[:, -1], bins=1000, alpha=0.8, label='Generated data')
    # plt.hist(MC_values[:, -1], bins=1000, alpha=0.8, label='Monte Carlo data')
    # plt.legend()
    # plt.show()
    # plt.close()


compare_distributions(model, sabr, N_samples=10000)
