import sys
import os
sys.path.append(os.getcwd())
from neuralSDE.Models.Discriminator import Discriminator
from neuralSDE.Models.NeuralLV import NeuralLV
from neuralSDE.Models.NeuralLSV import NeuralLSV
from neuralSDE.Models.NeuralSDE import NeuralSDE
from neuralSDE.WassersteinGAN.trainGAN import train_GAN
from neuralSDE.Utilities.MC import SABR_MC
import torch
import numpy as np
import pandas as pd
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
batch_size = 20000
epochs = 100000
N_simulations = 20 * batch_size
N_steps = 96
period_length = N_steps // n_maturities

# Monte Carlo test data
MC_samples_test = 200000
test_normal_variables = torch.randn(2, MC_samples_test, N_steps, requires_grad=False)
# We will use antithetic Brownian paths for testing
test_normal_variables = torch.cat([test_normal_variables, -test_normal_variables], 1)

if not os.path.exists('./neuralSDE/WassersteinGAN/Data/target_Wasserstein.pth.tar'):
    print('Simulating target paths for WassersteinGAN distance calculation')
    sabr = SABR_MC(F_0=F0, sigma_0=sigma_0, r=rfr, alpha=alpha, beta=beta, rho=rho, Time_horizon=Time_horizon,
                   N_steps=N_steps, N_simulations=N_simulations)
    target_numpy = sabr.simulate_paths()
    target = torch.tensor(target_numpy, dtype=torch.float32)
    torch.save(target, './neuralSDE/WassersteinGAN/Data/target_Wasserstein.pth.tar')
else:
    target = torch.load('./neuralSDE/WassersteinGAN/Data/target_Wasserstein.pth.tar')

if not os.path.exists('./neuralSDE/WassersteinGAN/Results/WassersteinGAN_NeuralLV.pth.tar'):
    with open("./neuralSDE/WassersteinGAN/Results/log_eval_NeuralLV.txt", "w") as f:
        f.write('epoch,loss\n')

    generatorLV = NeuralLV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                           N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                           S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                           num_layers=num_layers, layer_size=layer_size,
                           num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                           test_normal_variables=test_normal_variables[0])
    discriminatorLV = Discriminator(N_steps, num_layers, layer_size, device)
    print('Neural Local Volatility Model initiated')
    train_GAN(generatorLV, discriminatorLV, target, epochs, batch_size, device)
    print('Neural Local Volatility Model trained')
    torch.cuda.empty_cache()

if not os.path.exists('./neuralSDE/WassersteinGAN/Results/WassersteinGAN_NeuralLSV.pth.tar'):
    with open("./neuralSDE/WassersteinGAN/Results/log_eval_NeuralLSV.txt", "w") as f:
        f.write('epoch,loss\n')

    generatorLSV = NeuralLSV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                             N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                             S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                             num_layers=num_layers, layer_size=layer_size,
                             num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                             test_normal_variables=test_normal_variables)
    discriminatorLSV = Discriminator(N_steps, num_layers, layer_size, device)
    print('Neural Local Stochastic Volatility Model initiated')
    train_GAN(generatorLSV, discriminatorLSV, target, epochs, batch_size, device)
    print('Neural Local Stochastic Volatility Model trained')
    torch.cuda.empty_cache()

if not os.path.exists('./neuralSDE/WassersteinGAN/Results/WassersteinGAN_NeuralSDE.pth.tar'):
    with open("./neuralSDE/WassersteinGAN/Results/log_eval_NeuralSDE.txt", "w") as f:
        f.write('epoch,loss\n')

    generatorSDE = NeuralSDE(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                             N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                             S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                             num_layers=num_layers, layer_size=layer_size,
                             num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                             test_normal_variables=test_normal_variables)
    discriminatorSDE = Discriminator(N_steps, num_layers, layer_size, device)
    print('Neural Stochastic Differential Equation Model initiated')
    train_GAN(generatorSDE, discriminatorSDE, target, epochs, batch_size, device)
    print('Neural Stochastic Differential Equation Model trained')
