import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from Utilities.BaseNet import timegridNet


class NeuralLSV(nn.Module):
    def __init__(self, device, S0, N_simulations, N_steps, batch_size, test_normal_variables,
                 n_maturities, n_strikes, Time_horizon, rfr, period_length,
                 num_layers, layer_size,
                 num_layers_hedging, layer_size_hedging,
                 activation='leaky relu', diffusion_output_activation='softplus', output_activation='id',
                 dropout=0., use_hedging=True, use_batchnorm=True):
        super(NeuralLSV, self).__init__()
        n_S = 1
        n_V = 1
        sizes = [n_S + n_V + 1] + num_layers * [layer_size] + [n_S]
        sizes_b_V = [n_V] + num_layers * [layer_size] + [n_V]
        sizes_sigma_V = [n_V] + num_layers * [layer_size] + [n_V]
        activation_dict = {'relu': nn.ReLU(), 'leaky relu': nn.LeakyReLU(), 'id': nn.Identity(), 'tanh': nn.Tanh(), 'softplus': nn.Softplus()}

        F = activation_dict[activation]
        F_diff_output = activation_dict[diffusion_output_activation]
        F_output = activation_dict[output_activation]

        self.n_maturities = n_maturities
        self.n_strikes = n_strikes
        self.sigma_S = timegridNet(sizes=sizes, n_maturities=n_maturities, activation=F, output_activation=F_diff_output, dropout=dropout, use_batchnorm=use_batchnorm)
        self.b_V = timegridNet(sizes=sizes_b_V, n_maturities=n_maturities, activation=F, output_activation=F_output, dropout=dropout, use_batchnorm=use_batchnorm)
        self.sigma_V = timegridNet(sizes=sizes_sigma_V, n_maturities=n_maturities, activation=F, output_activation=F_diff_output, dropout=dropout, use_batchnorm=use_batchnorm)
        self.use_hedging = use_hedging
        if self.use_hedging:
            sizes_hedging = [n_S + 1] + num_layers_hedging * [layer_size_hedging] + [n_S * n_maturities * n_strikes]
            self.Hedging_Vanilla = timegridNet(sizes=sizes_hedging, n_maturities=n_maturities, activation=F, output_activation=F_output, dropout=dropout, use_batchnorm=use_batchnorm)
        self.rho = nn.Parameter(torch.tanh((2 * torch.rand(1) - 1)))
        self.S0 = torch.tensor(S0, dtype=torch.float)
        self.V0 = nn.Parameter(torch.sigmoid(torch.rand(1) - 3) * 0.5)
        self.N_simulations = N_simulations
        self.N_steps = N_steps
        self.batch_size = batch_size
        self.Time_horizon = Time_horizon
        self.dt = torch.tensor(Time_horizon / N_steps)
        self.rfr = torch.tensor(rfr)
        self.df = torch.exp(-self.rfr * self.dt)
        self.period_length = period_length
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.test_normal_variables = test_normal_variables

    def forward(self, maturities, strikes, test=False):  # torch version
        maturities = (maturities * self.N_steps / self.Time_horizon).astype(int).tolist()
        maturities_dict = dict(zip(maturities, range(self.n_maturities)))
        strikes = torch.tensor(strikes, device=self.device, dtype=torch.float).reshape(-1, 1)
        S0 = self.S0.to(self.device)
        V0 = self.V0.to(self.device)
        rho = self.rho.to(self.device)
        dt = self.dt.to(self.device)
        r = self.rfr.to(self.device)
        df = self.df.to(self.device)
        n_maturities = len(maturities)
        n_strikes = len(strikes)

        prices = torch.zeros(n_strikes, n_maturities, device=self.device, requires_grad=True)
        variance = torch.zeros(n_strikes, n_maturities, device=self.device, requires_grad=True)

        if test:
            S_prev = S0.repeat(1, self.test_normal_variables.shape[1])
            V_prev = V0.repeat(1, self.test_normal_variables.shape[1])
            ones = torch.ones(1, self.test_normal_variables.shape[1], dtype=torch.float)
            if self.use_hedging:
                hedging = torch.zeros(n_strikes, n_maturities, self.test_normal_variables.shape[1], device=self.device)
        else:
            S_prev = S0.repeat(1, self.batch_size)
            V_prev = V0.repeat(1, self.batch_size)
            ones = torch.ones(1, self.batch_size)
            if self.use_hedging:
                hedging = torch.zeros(n_strikes, n_maturities, self.batch_size, device=self.device)

        for i in range(1, self.N_steps + 1):
            idx = (i - 1) // self.period_length
            t = (i - 1) * dt * ones.to(self.device)
            X = torch.cat([t, S_prev, V_prev], 0).to(self.device)

            diffusion_S = self.sigma_S(idx, X)
            b_S = S_prev * r / (1 + abs(S_prev.detach() * r) * torch.sqrt(dt))
            sigma_S = S_prev * diffusion_S / (1 + abs(S_prev.detach() * diffusion_S.detach()) * torch.sqrt(dt))
            b_V = self.b_V(idx, V_prev)
            sigma_V = self.sigma_V(idx, V_prev)

            if test:
                NN1 = self.test_normal_variables[0, :, i - 1].to(self.device)
                NN2 = self.test_normal_variables[1, :, i - 1].to(self.device)
            else:
                NN1 = torch.randn(self.batch_size, device=self.device, requires_grad=False)
                NN2 = torch.randn(self.batch_size, device=self.device, requires_grad=False)
            dW = torch.sqrt(dt) * NN2
            dB = rho * dW + torch.sqrt(1 - rho ** 2) * torch.sqrt(dt) * NN1

            S_curr = S_prev + b_S * dt + sigma_S * dW
            V_curr = torch.clamp(V_prev + b_V * dt + sigma_V * dB, 0)
            if self.use_hedging:
                hedge = self.Hedging_Vanilla(idx, torch.cat([t, S_prev.detach()], 0)).reshape(n_strikes, n_maturities, -1)
                hedging += df ** (i - 1) * S_prev.detach() * diffusion_S.detach() * hedge * dW

            S_prev = S_curr
            V_prev = V_curr

            if i in maturities:
                table_index = maturities_dict[i]
                discount = df ** i
                simulated_prices = discount * torch.clamp(S_curr - strikes, 0)
                if self.use_hedging:
                    simulated_prices = simulated_prices - hedging[:, table_index]
                price = torch.zeros_like(prices)
                var = torch.zeros_like(prices)
                for j, strike in enumerate(strikes):
                    price[j, table_index] = simulated_prices[j].mean()
                    var[j, table_index] = simulated_prices[j].var()
                prices = prices + price
                variance = variance + var

        return prices, variance

    def simulate_paths(self, S_prev, V_prev, N_step):
        dt = self.dt
        r = self.rfr
        rho = self.rho

        if self.training:
            batch_size = self.batch_size
        else:
            batch_size = self.N_simulations
        if S_prev.ndim == 0:
            S_prev = S_prev.repeat(1, batch_size).to(self.device)
            V_prev = V_prev.repeat(1, batch_size).to(self.device)

        ones = torch.ones(1, batch_size, dtype=torch.float)
        idx = (N_step - 1) // self.period_length
        t = (N_step - 1) * dt * ones.to(self.device)
        X = torch.cat([t, S_prev, V_prev], 0).to(self.device)

        diffusion_S = self.sigma_S(idx, X)
        b_S = S_prev * r / (1 + abs(S_prev.detach() * r) * torch.sqrt(dt))
        sigma_S = S_prev * diffusion_S / (1 + abs(S_prev.detach() * diffusion_S.detach()) * torch.sqrt(dt))
        b_V = self.b_V(idx, V_prev)
        sigma_V = self.sigma_V(idx, V_prev)

        NN1 = torch.randn(self.batch_size, device=self.device, requires_grad=False)
        NN2 = torch.randn(self.batch_size, device=self.device, requires_grad=False)
        dW = torch.sqrt(dt) * NN2
        dB = rho * dW + torch.sqrt(1 - rho ** 2) * torch.sqrt(dt) * NN1

        S_curr = S_prev + b_S * dt + sigma_S * dW
        V_curr = torch.clamp(V_prev + b_V * dt + sigma_V * dB, 0)
        return S_curr, V_curr


def train(model, maturities, strikes, target, batch_size, epochs, threshold=2e-5):
    loss_fn = nn.MSELoss()

    model = model.to(model.device)
    networks_SDE = [model.sigma_S, model.b_V, model.sigma_V]
    parameters_SDE = list(model.sigma_S.parameters()) + list(model.b_V.parameters()) + list(model.sigma_V.parameters()) + [model.rho, model.V0]
    optimizer_SDE = optim.Adam(parameters_SDE, lr=1e-3)
    scheduler_SDE = optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500, 800], gamma=0.2, verbose=True)
    if model.use_hedging:
        optimizer_Hedging = optim.Adam(list(model.Hedging_Vanilla.parameters()), lr=1e-3)

    loss_val_best = 10
    itercount = 0
    LOSSES = []
    if model.use_hedging:
        VARIANCES = []
    for epoch in range(epochs):
        if model.use_hedging:
            optim_SDE = (epoch % 2 == 0)
            print(f'SDE parameters optimization: {optim_SDE}')

            if optim_SDE:
                for net in networks_SDE:
                    net.unfreeze()
                for param in parameters_SDE:
                    param.requires_grad_(True)
                model.Hedging_Vanilla.freeze()
            else:
                for net in networks_SDE:
                    net.freeze()
                for param in parameters_SDE:
                    param.requires_grad_(False)
                model.Hedging_Vanilla.unfreeze()

            for batch in range(0, 20 * batch_size, batch_size):
                optimizer_SDE.zero_grad()
                optimizer_Hedging.zero_grad()

                init_time = time.time()
                prices, prices_var = model(maturities, strikes, test=False)
                time_forward = time.time() - init_time
                itercount += 1

                if optim_SDE:
                    loss = loss_fn(prices, target)
                    init_time = time.time()
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters_SDE, 5)
                    time_backward = time.time() - init_time
                    optimizer_SDE.step()
                else:
                    loss_sample_var = prices_var.sum()
                    init_time = time.time()
                    loss = loss_sample_var
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.Hedging_Vanilla.parameters(), 5)
                    time_backward = time.time() - init_time
                    optimizer_Hedging.step()

                print(f'iteration: {itercount}, loss: {loss.item()}, {time_forward=}, {time_backward=}')

        else:
            for batch in range(0, 20 * batch_size, batch_size):
                optimizer_SDE.zero_grad()
                init_time = time.time()
                prices, _ = model(maturities, strikes, test=False)
                time_forward = time.time() - init_time
                itercount += 1
                loss = loss_fn(prices, target)
                init_time = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters_SDE, 5)
                time_backward = time.time() - init_time
                print(f'iteration: {itercount}, loss: {loss.item()}, {time_forward=}, {time_backward=}')
                optimizer_SDE.step()
        scheduler_SDE.step()

        with torch.no_grad():
            prices_mean, prices_var = model(maturities, strikes, test=True)
            print(f'Prices: \n{prices_mean}, \n Target: \n{target}, \n Variance: \n{prices_var}')

        MSE = loss_fn(prices_mean, target)
        loss_val = torch.sqrt(MSE)
        print(f'epoch={epoch}, loss={loss_val.item()}')
        with open("Results/log_eval_NeuralLSV.txt", "a") as f:
            f.write(f'{epoch},{loss_val.item()}\n')

        LOSSES.append(loss_val.item())

        if model.use_hedging:
            VARIANCES.append(prices_var.mean().item())
            if len(LOSSES) > 1 and len(VARIANCES) > 1:
                fig, axs = plt.subplots(1, 2)
                axs[0].plot(LOSSES, label='RMSE Error')
                axs[0].legend()
                axs[1].plot(VARIANCES, label='Mean price Variance')
                axs[1].legend()
                plt.savefig('Results/loss_and_var_NeuralLSV.png')
                plt.close()
        else:
            if len(LOSSES) > 1:
                plt.plot(LOSSES, label='RMSE Error')
                plt.legend()
                plt.savefig('Results/loss_neuralLSV.png')
                plt.close()

        if loss_val < loss_val_best:
            model_best = model
            loss_val_best = loss_val
            print(f'loss_val_best: {loss_val_best.item()}')
            filename = 'Results/NeuralLSV.pth.tar'
            checkpoint = {'state_dict': model.state_dict(), 'pred': prices_mean, 'target': target}
            torch.save(checkpoint, filename)

        if loss_val.item() < threshold:
            break

    return model_best


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    print(f'Using {device}')

    options = pd.read_csv('Data/Options_results.csv')
    options = options[options['Expiration_date'].isin([2.0])]
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
    epochs = 1000
    N_simulations = 20 * batch_size
    N_steps = 96
    period_length = N_steps // n_maturities

    # Monte Carlo test data
    MC_samples_test = 200000
    test_normal_variables = torch.randn(2, MC_samples_test, N_steps, requires_grad=False)
    # We will use antithetic Brownian paths for testing
    test_normal_variables = torch.cat([test_normal_variables, -test_normal_variables], 1)

    with open("Results/log_eval_NeuralLSV.txt", "w") as f:
        f.write('epoch,loss\n')

    model = NeuralLSV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                      N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                      S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                      num_layers=num_layers, layer_size=layer_size,
                      num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                      test_normal_variables=test_normal_variables)
    summary(model)
    print('Model initiated')
    train(model, maturities=maturities, strikes=strikes, target=target, batch_size=batch_size, epochs=epochs, threshold=2e-5)
    print('Model trained')
