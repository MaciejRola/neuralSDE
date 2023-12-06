import torch.nn as nn
import torch.nn.functional as F
from BaseNet import timegridNet
import torch.optim as optim
import torch
import pandas as pd
from MC import VanillaOption
torch.autograd.set_detect_anomaly(True)


def data_from_csv(path):
    options_csv = pd.read_csv(path, sep=',')
    options = [VanillaOption(row[0], row[1], row[2]) for row in zip(options_csv['Expiration_date'], options_csv['Strike'], options_csv['Call_or_Put'])]
    prices = torch.tensor(options_csv['Price'])
    return options, prices


options_train, prices_train = data_from_csv('Data/Options_train.csv')
options_test, prices_test = data_from_csv('Data/Options_test.csv')


class NeuralSDE(nn.Module):
    def __init__(self, device, n_S, n_V, S0, num_layers, layer_size, N_simulations, N_steps, n_maturities, Time_horizon, rfr, activation='relu', output_activation='id', dropout=0., use_batchnorm=True):
        super(NeuralSDE, self).__init__()
        sizes = [n_S + n_V + 1] + num_layers * [layer_size]

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'softmax':
            activation = nn.Softplus()

        if output_activation == 'id':
            output_activation = nn.Identity()
        elif output_activation == 'tanh':
            output_activation = nn.Tanh()
        self.n_S = n_S
        self.n_V = n_V
        self.n_maturities = n_maturities
        self.sigma_S = timegridNet(sizes + [n_S], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.b_V = timegridNet(sizes + [n_V], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.sigma_V = timegridNet(sizes + [n_V], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.Hedging_Vanilla = timegridNet(sizes + [n_S], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.Hedging_Exotics = None
        self.rho = nn.Parameter(2 * torch.rand(1) - 1)
        self.S0 = S0
        self.V0 = nn.Parameter(torch.rand(1) - 3)
        self.N_simulations = N_simulations
        self.N_steps = N_steps
        self.Time_horizon = Time_horizon
        self.dt = Time_horizon / N_steps
        self.rfr = rfr
        self.activation = activation
        self.output_activation = output_activation
        self.device = device
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

    def discounted_option_prices(self, S, hedging, options, batch_size):
        for option in options:
            option.timesteps_to_maturity(self.N_steps, self.Time_horizon)
        discounting = torch.exp(-self.rfr * torch.tensor([self.dt * i for i in range(self.N_steps + 1)])).reshape(1, -1, 1).repeat(self.n_S, 1, batch_size)
        discounting = discounting.to(self.device)
        price = torch.zeros((len(options), batch_size))
        hedge = torch.zeros((len(options), batch_size))
        price_mean = torch.zeros((len(options)))
        price_variance = torch.zeros((len(options)))
        for i, option in enumerate(options):
            maturity_idx = option.N_steps_T
            if isinstance(option, VanillaOption):
                if option.call_or_put == 'Call':
                    price[i] = discounting[:, maturity_idx, :] * torch.clamp(S[:, maturity_idx, :] - option.K, 0)
                else:
                    price[i] = discounting[:, maturity_idx, :] * torch.clamp(option.K - S[:, maturity_idx, :], 0)
                hedge[i] = hedging[:, maturity_idx, :]
                price_mean[i] = (price[i] - hedge[i]).mean()
                price_variance[i] = (price[i] - hedge[i]).var()
            else:
                if option.is_path_dependent:
                    price[i] = discounting[maturity_idx] * option.payoff(S)
                    hedge[i] = hedging[:, maturity_idx, :]
                else:
                    price[i] = discounting[maturity_idx] * option.payoff(S[:, maturity_idx, :])
                    hedge[i] = hedging[:, maturity_idx, :]
        return price_mean.float(), price_variance.float()

    def tamedEulerScheme(self, batch_size=None):  # torch version
        if not batch_size:
            batch_size = self.N_simulations
        S0 = torch.tensor(self.S0, device=self.device)
        V0 = (F.sigmoid(torch.tensor(self.V0, device=self.device)) * 0.5)
        rho = F.tanh(self.rho)
        dt = torch.tensor(self.Time_horizon / self.N_steps, device=self.device)
        n_S = self.n_S
        n_V = self.n_V
        ones = torch.ones((1, batch_size), device=self.device)
        S = torch.zeros((n_S, self.N_steps + 1, batch_size), device=self.device)
        V = torch.zeros((n_V, self.N_steps + 1, batch_size), device=self.device)
        hedging = torch.zeros((n_S, self.N_steps + 1, batch_size), device=self.device)
        S[:, 0, :] = S0
        V[:, 0, :] = V0
        r = torch.tensor(self.rfr, device=self.device)

        NN1 = torch.randn((self.N_steps, n_S, batch_size), device=self.device)
        NN2 = torch.randn((self.N_steps, n_V, batch_size), device=self.device)
        dW = torch.sqrt(dt) * NN2
        dB = rho * dW + torch.sqrt(1 - rho ** 2) * torch.sqrt(dt) * NN1

        for i in range(1, self.N_steps + 1):
            t = torch.tensor((i - 1) * self.dt, device=self.device)
            t_tensor = t * ones
            idx = 0
            S_prev = S[:, i - 1, :]
            V_prev = V[:, i - 1, :]
            hedging_prev = hedging[:, i - 1, :]
            X = torch.cat([t_tensor, S_prev, V_prev], 0)
            b_S = S_prev * r / (1 + abs(S_prev.detach() * r) * torch.sqrt(dt))
            sigma_S = self.sigma_S(idx, X) / (1 + abs(self.sigma_S(idx, X.detach())) * torch.sqrt(dt))
            b_V = self.b_V(idx, X) / (1 + abs(self.b_V(idx, X.detach())) * torch.sqrt(dt))
            sigma_V = self.sigma_V(idx, X) / (1 + abs(self.sigma_V(idx, X.detach())) * torch.sqrt(dt))
            S[:, i, :] = S_prev + b_S * dt + sigma_S * dB[i - 1, :, :]
            V[:, i, :] = torch.clamp(V_prev + b_V * dt + sigma_V * dW[i - 1, :, :], 0)
            hedging[:, i, :] = hedging_prev + torch.exp(-self.rfr * t) * S_prev.detach() * sigma_S.detach() * self.Hedging_Vanilla(idx, torch.cat([t_tensor, S_prev.detach(), V_prev.detach()], 0))
        return S, V, hedging

    def forward(self, options, batch_size=None):
        if not batch_size:
            batch_size = self.N_simulations
        S, V, hedging = self.tamedEulerScheme(batch_size)
        prices = self.discounted_option_prices(S, hedging, options, batch_size)
        return prices


def train(model, target_prices, options, batch_size, epochs, threshold):
    loss_fn = nn.MSELoss()
    model = model.to(model.device)

    target_prices = target_prices.to(model.device).float()
    networks_SDE = [model.sigma_S, model.b_V, model.sigma_V]
    parameters_SDE = [model.rho, model.V0]

    optimizer_SDE = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_Hedging = optim.Adam(model.parameters(), lr=1e-3)

    loss_val_best = 10
    best_model = None
    for epoch in range(epochs):
        optim_SDE = (epoch % 2 == 0)

        for batch in range(0, 20 * batch_size, batch_size):
            optimizer_SDE.zero_grad()
            optimizer_Hedging.zero_grad()
            prices_mean, prices_var = model(options, batch_size)
            prices_mean = prices_mean.to(model.device)
            prices_var = prices_var.to(model.device)
            if optim_SDE:
                for net in networks_SDE:
                    net.unfreeze()
                for param in parameters_SDE:
                    param.requires_grad_(True)
                model.Hedging_Vanilla.freeze()

                loss = F.mse_loss(prices_mean, target_prices)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters_SDE, 2)
                optimizer_SDE.step()

            else:
                for net in networks_SDE:
                    net.freeze()
                for param in parameters_SDE:
                    param.requires_grad_(True)
                model.Hedging_Vanilla.unfreeze()

                loss_sample_var = prices_var.sum()
                loss = loss_sample_var
                loss.backward()
                nn.utils.clip_grad_norm_(model.Hedging_Vanilla.parameters(), 2)
                optimizer_Hedging.step()

        with torch.no_grad():
            prices_mean, prices_var = model(options)
            prices_mean = prices_mean.to(model.device)
            prices_var = prices_var.to(model.device)
            print(f'{prices_mean=}, {prices_var=}')

        MSE = loss_fn(prices_mean, target_prices)
        loss_val = torch.sqrt(MSE)
        print(loss_val)

        # if loss_val < loss_val_best:
        #     best_model = model
        #     loss_val_best = loss_val
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "prices_mean": prices_mean,
        #         "target": target_prices,
        #     }
        #     torch.save(checkpoint, "best_model.pth")
        # if loss_val.item() < threshold:
        #     break
    return best_model


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'
#device = 'cpu'
model = NeuralSDE(device=device, n_S=1, n_V=1, S0=100, num_layers=2, layer_size=64, N_simulations=200000, N_steps=96, n_maturities=1, Time_horizon=2, rfr=0.05, dropout=0.1, use_batchnorm=False)
print('Model initiated')
train(model, target_prices=prices_train, options=options_train, batch_size=40000, epochs=1000, threshold=2e-5)
print('Model trained')
