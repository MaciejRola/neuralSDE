import torch
import torch.nn as nn
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
