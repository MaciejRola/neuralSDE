import torch.nn as nn
from BaseNet import timegridNet
import torch.optim as optim
import torch
import pandas as pd
from MC import VanillaOption
from torchinfo import summary
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)


def data_from_csv(path):
    options_csv = pd.read_csv(path, sep=',')
    options = [VanillaOption(row[0], row[1], row[2]) for row in zip(options_csv['Expiration_date'], options_csv['Strike'], options_csv['Call_or_Put'])]
    prices = torch.tensor(options_csv['Price'])
    return options, prices


#options_train, prices_train = data_from_csv('Data/Options_train.csv')
#options_test, prices_test = data_from_csv('Data/Options_test.csv')
options = pd.read_csv('Data/Options_results.csv')


class NeuralSDE(nn.Module):
    def __init__(self, device, S0, num_layers, num_layers_hedging, layer_size, layer_size_hedging, N_simulations, N_steps, n_maturities, n_strikes, Time_horizon, rfr, period_length, n_S=1, n_V=1, activation='relu', output_activation='id', dropout=0., use_batchnorm=True):
        super(NeuralSDE, self).__init__()
        sizes = [n_S + n_V + 1] + num_layers * [layer_size]
        sizes_hedging = [n_S + 1] + num_layers_hedging * [layer_size_hedging]
        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'softmax':
            activation = nn.Softplus()

        if output_activation == 'id':
            output_activation = nn.Identity()
        elif output_activation == 'tanh':
            output_activation = nn.Tanh()

        n_S = 1
        n_V = 1
        self.n_maturities = n_maturities
        self.n_strikes = n_strikes
        self.maturities_idx = [(i + 1) * period_length for i in range(n_maturities)]
        self.sigma_S = timegridNet(sizes + [n_S], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.b_V = timegridNet(sizes + [n_V], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.sigma_V = timegridNet(sizes + [n_V], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.Hedging_Vanilla = timegridNet(sizes_hedging + [n_S * n_maturities * n_strikes], n_maturities, activation, output_activation, dropout, use_batchnorm)
        self.Hedging_Exotics = None
        self.rho = nn.Parameter(torch.tanh((2 * torch.rand(1) - 1)))
        self.S0 = torch.tensor(S0)
        self.V0 = nn.Parameter(torch.sigmoid(torch.rand(1) - 3)*0.5)
        self.N_simulations = N_simulations
        self.N_steps = N_steps
        self.Time_horizon = Time_horizon
        self.dt = torch.tensor(Time_horizon / N_steps)
        self.rfr = torch.tensor(rfr)
        self.period_length = period_length
        self.activation = activation
        self.output_activation = output_activation
        self.device = device
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

    def forward(self, options, batch_size=None):  # torch version
        maturities = (options['Expiration_date'].unique() * self.N_steps / self.Time_horizon).astype(int).tolist()
        maturities_dict = dict(zip(maturities, range(self.n_maturities)))
        strikes = torch.tensor(options['Strike'].unique(), device=self.device, dtype=torch.float).reshape(-1, 1)
        if not batch_size:
            batch_size = self.N_simulations
        S0 = self.S0
        V0 = self.V0
        rho = self.rho
        dt = self.dt
        n_maturities = self.n_maturities
        n_strikes = self.n_strikes

        assert n_maturities * n_strikes == len(options)

        prices = torch.zeros(n_strikes, n_maturities, device=self.device, requires_grad=True)
        variance = torch.zeros(n_strikes, n_maturities, device=self.device, requires_grad=True)
        S_path = torch.zeros(self.N_steps + 1, batch_size, requires_grad=False)
        hedging = torch.zeros(n_strikes, n_maturities, batch_size, device=self.device)
        S_path[0, :] = S0.detach()
        r = self.rfr
        df = torch.exp(-r * dt)
        S_prev = S0.to(self.device).repeat(1, batch_size)
        V_prev = V0.to(self.device).repeat(1, batch_size)
        for i in range(1, self.N_steps + 1):
            t = (i - 1) * dt.repeat(1, batch_size).to(self.device)
            S_prev = S_prev.to(self.device)
            V_prev = V_prev.to(self.device)

            idx = (i - 1) // self.period_length

            X = torch.cat([t, S_prev, V_prev], 0)

            b_S = S_prev * r / (1 + abs(S_prev.detach() * r) * torch.sqrt(dt))
            sigma_S = self.sigma_S(idx, X) / (1 + abs(self.sigma_S(idx, X.detach())) * torch.sqrt(dt))
            b_V = self.b_V(idx, X) / (1 + abs(self.b_V(idx, X.detach())) * torch.sqrt(dt))
            sigma_V = self.sigma_V(idx, X) / (1 + abs(self.sigma_V(idx, X.detach())) * torch.sqrt(dt))

            NN1 = torch.randn(batch_size, device=self.device, requires_grad=False)
            NN2 = torch.randn(batch_size, device=self.device, requires_grad=False)
            dW = torch.sqrt(dt) * NN2
            dB = rho * dW + torch.sqrt(1 - rho ** 2) * torch.sqrt(dt) * NN1

            S_curr = S_prev + b_S * dt + sigma_S * dB
            V_curr = torch.clamp(V_prev + b_V * dt + sigma_V * dW, 0)
            hedge = self.Hedging_Vanilla(idx, torch.cat([t, S_prev.detach()], 0)).reshape(n_strikes, n_maturities, -1)
            hedging += df * S_prev.detach() * sigma_S.detach() * hedge * dW

            S_path[i, :] = S_curr.detach()

            S_prev = S_curr
            V_prev = V_curr

            if i in maturities:
                table_index = maturities_dict[i]
                discount = df ** i
                simulated_prices = discount * torch.clamp(S_curr - strikes, 0) - hedging[:, table_index]
                price = torch.zeros_like(prices)
                var = torch.zeros_like(prices)
                for j, strike in enumerate(strikes):
                    price[j, table_index] = simulated_prices[j].mean()
                    var[j, table_index] = simulated_prices[j].var()
                prices = prices + price
                variance = variance + var

        return prices, variance


def train(model, options, batch_size, epochs, threshold):
    loss_fn = nn.MSELoss()
    model = model.to(model.device)
    target_prices = torch.tensor(pd.pivot_table(options, values='Price', index='Strike', columns='Expiration_date').to_numpy(), requires_grad=True, device=model.device, dtype=torch.float)

    networks_SDE = [model.sigma_S, model.b_V, model.sigma_V]
    parameters_SDE = [model.rho, model.V0]

    optimizer_SDE = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_Hedging = optim.Adam(model.parameters(), lr=1e-3)

    loss_val_best = 10
    best_model = None
    for epoch in range(epochs):
        optim_SDE = (epoch % 2 == 0)
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
        for batch in range(0, model.N_simulations, batch_size):
            optimizer_SDE.zero_grad()
            optimizer_Hedging.zero_grad()
            prices, prices_var = model(options, batch_size)
            prices = prices.to(model.device)
            prices_var = prices_var.to(model.device)
            if optim_SDE:
                loss = loss_fn(prices, target_prices)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters_SDE, 5)
                optimizer_SDE.step()
            else:
                loss_sample_var = prices_var.sum()
                loss = loss_sample_var
                loss.backward()
                nn.utils.clip_grad_norm_(model.Hedging_Vanilla.parameters(), 5)
                optimizer_Hedging.step()

        with torch.no_grad():
            prices_mean, prices_var = model(options)
            prices_mean = prices_mean.to(model.device)
            prices_var = prices_var.to(model.device)
            print(f'{prices_mean=}, {prices_var=}')

        MSE = loss_fn(prices_mean, target_prices)
        loss_val = torch.sqrt(MSE)
        print(loss_val)
        LOSSES.append(loss_val.item())
        if len(LOSSES) > 1:
            plt.plot(LOSSES)
            plt.savefig('losses.png')
        if loss_val < loss_val_best:
            best_model = model
            loss_val_best = loss_val
            checkpoint = {
                "state_dict": model.state_dict(),
                "prices_mean": prices_mean,
                "target": target_prices,
            }
            torch.save(checkpoint, "best_model_no_bn.pth")
        if loss_val.item() < threshold:
            break
    return best_model


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = 'cpu'
print(f'Using {device}')
n_S = 1
n_V = 1
S0 = 100
num_layers = 3
num_layers_hedging = 4
layer_size = 32
layer_size_hedging = 64
N_simulations = 200000
N_steps = 96
n_maturities = 4
n_strikes = 201
Time_horizon = 2
rfr = 0.05
dropout = 0.0
use_batchnorm = True
LOSSES = []
period_length = N_steps // n_maturities
model = NeuralSDE(device=device, S0=S0, num_layers=num_layers, num_layers_hedging=num_layers_hedging, layer_size=layer_size, layer_size_hedging=layer_size_hedging,
                  N_simulations=N_simulations, N_steps=N_steps, n_maturities=n_maturities, n_strikes=n_strikes, Time_horizon=Time_horizon, rfr=rfr, period_length=period_length,
                  activation='relu', output_activation='id', dropout=dropout, use_batchnorm=use_batchnorm)
summary(model)
print('Model initiated')
train(model, options=options, batch_size=40000, epochs=1000, threshold=2e-5)
print('Model trained')
