import torch
import torch.nn as nn
from neuralSDE.Utilities.BaseNet import timegridNet


class NeuralLV(nn.Module):
    def __init__(self, device, S0, N_simulations, N_steps, batch_size, test_normal_variables,
                 n_maturities, n_strikes, Time_horizon, rfr, period_length,
                 num_layers, layer_size,
                 num_layers_hedging, layer_size_hedging,
                 activation='leaky relu', diffusion_output_activation='softplus', output_activation='id',
                 dropout=0., use_hedging=True, use_batchnorm=True):
        super(NeuralLV, self).__init__()
        n_S = 1
        sizes = [n_S + 1] + num_layers * [layer_size] + [n_S]
        activation_dict = {'relu': nn.ReLU(), 'leaky relu': nn.LeakyReLU(), 'id': nn.Identity(), 'tanh': nn.Tanh(), 'softplus': nn.Softplus()}

        F = activation_dict[activation]
        F_output = activation_dict[output_activation]
        F_diff_output = activation_dict[diffusion_output_activation]
        self.n_maturities = n_maturities
        self.n_strikes = n_strikes
        self.leverage = timegridNet(sizes=sizes, n_maturities=n_maturities, activation=F, output_activation=F_diff_output, dropout=dropout, use_batchnorm=use_batchnorm)
        self.use_hedging = use_hedging
        if self.use_hedging:
            sizes_hedging = [n_S + 1] + num_layers_hedging * [layer_size_hedging]
            self.Hedging_Vanilla = timegridNet(sizes=sizes_hedging + [n_S * n_maturities * n_strikes], n_maturities=n_maturities, activation=F, output_activation=F_output, dropout=dropout,
                                               use_batchnorm=use_batchnorm)
        self.S0 = torch.tensor(S0, dtype=torch.float)
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
        dt = self.dt.to(self.device)
        r = self.rfr.to(self.device)
        df = self.df.to(self.device)
        n_maturities = len(maturities)
        n_strikes = len(strikes)

        prices = torch.zeros(n_strikes, n_maturities, device=self.device, requires_grad=True)
        variance = torch.zeros(n_strikes, n_maturities, device=self.device, requires_grad=True)

        if test:
            S_prev = S0.repeat(1, self.test_normal_variables.shape[0])
            ones = torch.ones(1, self.test_normal_variables.shape[0], dtype=torch.float)
            if self.use_hedging:
                hedging = torch.zeros(n_strikes, n_maturities, self.test_normal_variables.shape[0], device=self.device)
        else:
            S_prev = S0.repeat(1, self.batch_size)
            ones = torch.ones(1, self.batch_size)
            if self.use_hedging:
                hedging = torch.zeros(n_strikes, n_maturities, self.batch_size, device=self.device)

        for i in range(1, self.N_steps + 1):
            idx = (i - 1) // self.period_length
            t = (i - 1) * dt * ones.to(self.device)
            X = torch.cat([t, S_prev], 0).to(self.device)

            leverage = self.leverage(idx, X)
            drift = S_prev * r / (1 + abs(S_prev.detach() * r) * torch.sqrt(dt))
            diffusion = S_prev * leverage / (1 + abs(S_prev.detach() * leverage.detach()) * torch.sqrt(dt))

            if test:
                NN = self.test_normal_variables[:, i - 1].to(self.device)
            else:
                NN = torch.randn(self.batch_size, device=self.device, requires_grad=False)
            dW = torch.sqrt(dt) * NN

            S_curr = S_prev + drift * dt + diffusion * dW
            if self.use_hedging:
                hedge = self.Hedging_Vanilla(idx, torch.cat([t, S_prev.detach()], 0)).reshape(n_strikes, n_maturities, -1)
                hedging += df ** (i - 1) * S_prev.detach() * diffusion.detach() * hedge * dW

            S_prev = S_curr

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

    def simulate_values(self, S_prev, N_step, N_simulations=None):
        dt = self.dt
        r = self.rfr
        if self.training:
            batch_size = self.batch_size
        elif N_simulations is not None:
            batch_size = N_simulations
        else:
            batch_size = self.N_simulations
        if S_prev.ndim == 0:
            S_prev = S_prev.repeat(1, batch_size).to(self.device)

        ones = torch.ones(1, batch_size, dtype=torch.float)
        idx = (N_step - 1) // self.period_length
        t = (N_step - 1) * dt * ones.to(self.device)
        X = torch.cat([t, S_prev], 0).to(self.device)

        leverage = self.leverage(idx, X)
        drift = S_prev * r / (1 + abs(S_prev.detach() * r) * torch.sqrt(dt))
        diffusion = leverage / (1 + abs(leverage.detach()) * torch.sqrt(dt))
        NN = torch.randn(batch_size, device=self.device, requires_grad=False)
        dW = torch.sqrt(dt) * NN
        S_curr = S_prev + drift * dt + diffusion * dW
        return S_curr

    def simulate_paths(self, batch_size):
        S = self.S0.repeat(1, batch_size).to(self.device)
        modelling_vol = hasattr(self, 'V0')
        if modelling_vol:
            V = self.V0.repeat(1, batch_size).to(self.device)
        paths = torch.zeros(batch_size, self.N_steps + 1, device=self.device)
        paths[:, 0] = S[0, :]
        for step in range(1, self.N_steps + 1):
            if modelling_vol:
                S, V = self.simulate_values(S, V, step)
            else:
                S = self.simulate_values(S, step)
            paths[..., step] = S
        return paths
